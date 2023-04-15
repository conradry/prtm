# Copyright 2021 (c) Mikita Misiura
#
# This code is part of DLPacker. If you use it in your work, please cite the
# following paper:
# 
# @article {Misiura2021.05.23.445347,
#     author = {Misiura, Mikita and Shroff, Raghav and Thyer, Ross and Kolomeisky, Anatoly},
#     title = {DLPacker: Deep Learning for Prediction of Amino Acid Side Chain Conformations in Proteins},
#     elocation-id = {2021.05.23.445347},
#     year = {2021},
#     doi = {10.1101/2021.05.23.445347},
#     publisher = {Cold Spring Harbor Laboratory},
#     URL = {https://www.biorxiv.org/content/early/2021/05/25/2021.05.23.445347},
#     eprint = {https://www.biorxiv.org/content/early/2021/05/25/2021.05.23.445347.full.pdf},
#     journal = {bioRxiv}
# }
# 
# Licensed under the MIT License:
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

import time, os, re

import numpy as np

import tensorflow as tf
import tensorflow.keras as K

from collections import defaultdict

# do not change any of these
THE20 = {'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'GLN': 5,\
         'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9, 'LEU': 10, 'LYS': 11,\
         'MET': 12, 'PHE': 13, 'PRO': 14, 'SER': 15, 'THR': 16,\
         'TRP': 17, 'TYR': 18, 'VAL': 19}
SCH_ATOMS = {'ALA': 1, 'ARG': 7, 'ASN': 4, 'ASP': 4, 'CYS': 2, 'GLN': 5,\
             'GLU': 5, 'GLY': 0, 'HIS': 6, 'ILE': 4, 'LEU': 4, 'LYS': 5,\
             'MET': 4, 'PHE': 7, 'PRO': 3, 'SER': 2, 'THR': 3,\
             'TRP': 10, 'TYR': 8, 'VAL': 3}
BB_ATOMS = ['C', 'CA', 'N', 'O']
SIDE_CHAINS = {'MET': ['CB', 'CE', 'CG', 'SD'],
               'ILE': ['CB', 'CD1', 'CG1', 'CG2'],
               'LEU': ['CB', 'CD1', 'CD2', 'CG'],
               'VAL': ['CB', 'CG1', 'CG2'],
               'THR': ['CB', 'CG2', 'OG1'],
               'ALA': ['CB'],
               'ARG': ['CB', 'CD', 'CG', 'CZ', 'NE', 'NH1', 'NH2'],
               'SER': ['CB', 'OG'],
               'LYS': ['CB', 'CD', 'CE', 'CG', 'NZ'],
               'HIS': ['CB', 'CD2', 'CE1', 'CG', 'ND1', 'NE2'],
               'GLU': ['CB', 'CD', 'CG', 'OE1', 'OE2'],
               'ASP': ['CB', 'CG', 'OD1', 'OD2'],
               'PRO': ['CB', 'CD', 'CG'],
               'GLN': ['CB', 'CD', 'CG', 'NE2', 'OE1'],
               'TYR': ['CB', 'CD1', 'CD2', 'CE1', 'CE2', 'CG', 'CZ', 'OH'],
               'TRP': ['CB', 'CD1', 'CD2', 'CE2', 'CE3', 'CG', 'CH2', 'CZ2', 'CZ3', 'NE1'],
               'CYS': ['CB', 'SG'],
               'ASN': ['CB', 'CG', 'ND2', 'OD1'],
               'PHE': ['CB', 'CD1', 'CD2', 'CE1', 'CE2', 'CG', 'CZ']}
BOX_SIZE = 10
GRID_SIZE = 40
SIGMA = 0.65

class DLPModel():
    # This class represents DNN model we used in this work
    # If you just want to use the pre-trained weights that
    # we've published, then there is nothing you will ever
    # need to change here
    def __init__(self, grid_size:int = 40, nres:int = 6, num_channels:int = 27,\
                       batch_size:int = 32, lr:float = 1e-4, width:int = 128):
        self.width = width # base number of channels
        self.lr = lr # learning rate
        self.nres = nres # number of residual layers
        self.grid_size = grid_size # grid size
        self.num_channels = num_channels # number of input channels
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(lr)
        self.data_gen = DataGenerator(self.batch_size, folder = './BOXES_TRAIN/')
        self.val_gen = DataGenerator(self.batch_size, folder = './BOXES_VAL/')
        
        self.model = self.model()
        
        self.loss_history = {'mae': [], 'roi': []}
        self.ema = 0.999 # for loss history smoothing
        
    def __str__(self):
        print('3D CNN Model\nLR: {}, BATCH SIZE: {}\n'.format(self.lr, self.batch_size))
        self.model.summary(line_length = 110)
        return 'lol'
    
    def load_model(self, weights:str, history:str = ''):
        self.model.load_weights(weights + '.h5')
        if history:
            with open(history + '.pkl', 'rb') as h:
                self.loss_history = pickle.load(h)
    
    def save_model(self, weights:str, history:str = ''):
        self.model.save_weights(weights + '.h5')
        if history:
            with open(history + '.pkl', 'wb') as f:
                pickle.dump(self.loss_history, f)
    
    def train(self, epochs:int):
        for e in range(epochs):
            start = time.time()
            for i, (x, y, labels) in enumerate(self.data_gen):
                with tf.GradientTape() as tape:
                    out = self.model([x, labels])
                    l = self.loss(x[..., :4], y, labels, out)
                    
                    grads = tape.gradient(l, self.model.trainable_variables)
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                end = time.time()
                
                print('Epoch: %d/%d' % (e + 1, epochs), 'Iteration:', i,\
                      'Losses [MAE, ROI]: [%.3e, %.3e] | Time per step: %.2f s'\
                       % (self.loss_history['mae'][-1],\
                          self.loss_history['roi'][-1],\
                          end - start), end = '\r')
                
                start = time.time()
                if i % 10000 == 0: self.model.save('backup')
    
    def validate(self):
        count = 0
        maes = []
        rois = []
        for i, (x, y, labels) in enumerate(self.val_gen):
            if i > 0: print('Batch:', i, np.mean(rois), end = '\r')
            out = self.model([x, labels])
            x = x[..., :4]
            mae = tf.reduce_mean(tf.math.abs(y - out))
            mask = np.array(x != y, dtype = np.float32)
            roi = tf.reduce_mean(tf.math.abs(y - out) * mask) * 100
            
            maes.append(mae)
            rois.append(roi)
        
        print('MAE:', np.mean(maes), 'ROI:', np.mean(rois))
                    
    def loss(self, x, y, labels, out):
        mae = tf.reduce_mean(tf.math.abs(y - out))
        
        mask = np.array(x != y, dtype = np.float32)
        roi = tf.reduce_mean(tf.math.abs(y - out) * mask) * 100
        
        if self.loss_history['mae']:
            mae_ema = mae.numpy() * (1 - self.ema) + self.ema * self.loss_history['mae'][-1]
            self.loss_history['mae'].append(mae_ema)
            roi_ema = roi.numpy() * (1 - self.ema) + self.ema * self.loss_history['roi'][-1]
            self.loss_history['roi'].append(roi_ema)
        else:
            self.loss_history['mae'].append(mae.numpy())
            self.loss_history['roi'].append(roi.numpy())
        
        return roi + mae
    
    def model(self):
        width = self.width
        inp = K.layers.Input(shape = (self.grid_size, self.grid_size, self.grid_size, self.num_channels))
        labels = K.layers.Input(shape = (20,))
        
        fc = K.layers.Dense(self.grid_size * self.grid_size * self.grid_size, activation = 'relu')(labels)
        fc = tf.reshape(fc, shape = (-1, self.grid_size, self.grid_size, self.grid_size, 1))
        
        l0 = K.layers.Concatenate(axis = -1)([inp, fc])
        
        
        def res_identity(x, f1, f2):
            x_in = x
            x = K.layers.Conv3D(f1, 1, strides = 1, padding = 'valid', activation = 'relu')(x)
            x = K.layers.Conv3D(f1, 3, strides = 1, padding = 'same',  activation = 'relu')(x)
            x = K.layers.Conv3D(f2, 1, strides = 1, padding = 'valid')(x)
            x = K.layers.Add()([x, x_in])
            return K.layers.Activation('relu')(x)
        
        l1 = K.layers.Conv3D(1 * width, 3, padding = 'same', strides = 2, activation = 'relu')(l0)
        l2 = K.layers.Conv3D(2 * width, 3, padding = 'same', strides = 2, activation = 'relu')(l1)
        l3 = K.layers.Conv3D(4 * width, 3, padding = 'same', strides = 1, activation = 'relu')(l2)
        
        for _ in range(self.nres):
            l3 = res_identity(l3, 2 * width, 4 * width)
        
        l = K.layers.Concatenate(axis = -1)([l3, l2])
        l = K.layers.Conv3D(4 * width, 3, padding = 'same')(l)
        l = K.layers.LeakyReLU(alpha = 0.2)(l)

        l = K.layers.UpSampling3D(size = 2)(l)
        l = K.layers.Concatenate(axis = -1)([l, l1])
        l = K.layers.Conv3D(2 * width, 3, padding = 'same')(l)
        l = K.layers.LeakyReLU(alpha = 0.2)(l)

        l = K.layers.UpSampling3D(size = 2)(l)
        l = K.layers.Concatenate(axis = -1)([l, inp])
        l = K.layers.Conv3D(4, 3, padding = 'same')(l)
        
        l = l + inp[..., :4]
        
        return K.Model(inputs = [inp, labels], outputs = l, name = 'Generator')


class InputBoxReader():
    # Input generator creating 26 input channels for the DNN.
    # Takes in a special dictionary containing information about microenvironment
    # or a filename containing such a dictionary.
    # This version puts backbone atoms for each amino acid into separate channel
    # as well as into element channels.
    # Side chain atoms only go into element channels.
    # The structure is like this:
    #     Channels 1-4 - CNOS
    #     Channel  5 - all other elements
    #     Channel  6 - charges
    #     Channels 7-26 - backbones of amino acids, each channel for one AA
    # Output tensor has four element chanels (CNOS)
    def __init__(self, remove_sidechains:str = 'none',\
                       from_dict:bool = True, include_water:bool = False,\
                       charges_filename:str = 'charges.rtp'):
        self.grid_size = GRID_SIZE
        self.grid_spacing = BOX_SIZE * 2 / GRID_SIZE # grid step
        self.offset = 10 * GRID_SIZE // 40 # to include atoms on the border
        self.total_size = GRID_SIZE + 2 * self.offset
        self.remove_sidechains = remove_sidechains
        self.from_dict = from_dict # choose dict or filename as input
        self.include_water = include_water

        assert self.remove_sidechains in ['all', 'random', 'none']

        # preparing the kernel and grid
        size = round(SIGMA * 4) # kernel size
        self.grid = np.mgrid[-size:size+self.grid_spacing:self.grid_spacing,\
                             -size:size+self.grid_spacing:self.grid_spacing,\
                             -size:size+self.grid_spacing:self.grid_spacing]

        # defining a kernel
        kernel = np.exp(-np.sum(self.grid * self.grid, axis = 0) / SIGMA**2 / 2) 
        kernel /= (np.sqrt(2 * np.pi) * SIGMA)
        self.kernel = kernel[1:-1, 1:-1, 1:-1]
        self.norm = np.sum(self.kernel)
        
        # read in the charges from special file
        self.charges = defaultdict(lambda: 0) # output 0 if the key is absent
        with open(charges_filename, 'r') as f:
            for line in f:
                if line[0] == '[' or line[0] == ' ':
                    if re.match('\A\[ .{1,3} \]\Z', line[:-1]):
                        key = re.match('\A\[ (.{1,3}) \]\Z', line[:-1])[1]
                        self.charges[key] = defaultdict(lambda: 0)
                    else:
                        l = re.split(r' +', line[:-1])
                        self.charges[key][l[1]] = float(l[3])
    
    def __call__(self, box:[str, dict]):
        # input is either a dictionary or a filename with a dictionary stored in it
        if not self.from_dict:
            box = np.load(box, allow_pickle = True)
            box = box['arr_0'].item()

        # list of all amino acids except for target
        # in the end we want it to contain all amino acids
        # whose side chains we want to see in the input
        amino_acids = set(box['resids'])
        amino_acids.remove(int(box['target']['id']))
        
        if self.remove_sidechains == 'random':
            # if we choose to randomly remove amino acid sidechains
            # then we have 50% chance to remove them all
            # and 50% chance to remove random fraction of them
            if np.random.rand() < 0.25:
                p = np.random.rand()
                amino_acids = set([a for a in amino_acids if np.random.rand() < p])
            else:
                amino_acids = set()
        elif self.remove_sidechains == 'all':
            amino_acids = set()
        
        x  = np.zeros([self.total_size, self.total_size, self.total_size, 27])
        y  = np.zeros([self.total_size, self.total_size, self.total_size, 4])
        
        centers = (np.array(box['positions']) + BOX_SIZE) / self.grid_spacing
        centers += self.offset
        cr = np.round(centers).astype(np.int32)
        offsets = cr - centers
        offsets = offsets[:, :, None, None, None]
        
        i0 = self.kernel.shape[0] // 2
        i1 = self.kernel.shape[0] - i0
        
        for ind, a in enumerate(box['types']):
            if box['resnames'][ind] != 'HOH' or self.include_water:
                # defines fine position of the kernel
                dist = self.grid + offsets[ind] * self.grid_spacing
                kernel = np.exp(-np.sum(dist * dist, axis = 0) / SIGMA**2 / 2)
                kernel = kernel[1:-1, 1:-1, 1:-1] * self.norm / np.sum(kernel)

                # defining indeces to put atom into
                xa, xb = cr[ind][0]-i0, cr[ind][0]+i1
                ya, yb = cr[ind][1]-i0, cr[ind][1]+i1
                za, zb = cr[ind][2]-i0, cr[ind][2]+i1

                # define the channel for the atom
                if a == 'C': ch = 0
                elif a == 'N': ch = 1
                elif a == 'O': ch = 2
                elif a == 'S': ch = 3
                else: ch = 4

                aa = box['resnames'][ind] # amino acid
                an = box['names'][ind]    # atom name

                # filling in input element channels,
                # charge channel and all of output channels

                # check if the atom is in target side chain
                if ind in box['target']['atomids']:
                    # target atoms only go into output
                    if ch != 4: y[xa:xb, ya:yb, za:zb, ch] += kernel
                elif an in BB_ATOMS or box['resids'][ind] in amino_acids:
                    # otherwise, the atom goes into input
                    # element channels
                    x[xa:xb, ya:yb, za:zb, ch] += kernel
                    # all CNOS atoms also go into output
                    if ch != 4: y[xa:xb, ya:yb, za:zb, ch] += kernel
                    # add charges as same kernels multiplied
                    # by partial charge value
                    if aa in self.charges:
                        # if charge value is known, use it
                        charge = kernel * self.charges[aa][an]
                        x[xa:xb, ya:yb, za:zb, 5] += kernel * self.charges[aa][an]
                    else:
                        # otherwise use default values
                        charge = kernel * self.charges['RST'][an[:1]]
                        x[xa:xb, ya:yb, za:zb, 5] += charge
                # filling in amino acid backbone channels
                if an in BB_ATOMS:
                    if aa in THE20:
                        x[xa:xb, ya:yb, za:zb, 6 + THE20[aa]] += kernel
                    else:
                        x[xa:xb, ya:yb, za:zb, 6 + 20] += kernel
            b = self.offset
        return x[b:-b, b:-b, b:-b, :], y[b:-b, b:-b, b:-b, :]


class DataGenerator(tf.data.Dataset):
    # Pretty standard data denerator based on tf.data.Dataset
    def _generator(num_channels:int, grid_size:int, randomize:bool,\
                   folder:str, remove_sidechains:str):
        folder = folder.decode('utf-8')
        remove_sidechains = remove_sidechains.decode('utf-8')
        # The target file structure is as folows:
        # withing the `folder` there are multiple subfolders named by PDB codes
        # and each subfolder contains .npz files with dictionaries stored
        # and named like this example: 1a2z_C_PHE_179.npz
        # so the final filename is like this: `folder`/1a2z/1a2z_C_PHE_179.npz
        files = []
        for f in os.listdir(folder):
            ff = os.listdir(folder + f)
            for file in ff:
                files.append(folder + f + '/' + file)
        
        if randomize: np.random.shuffle(files)

        input_reader = InputBoxReader(remove_sidechains = remove_sidechains)
        
        x = np.zeros((grid_size, grid_size, grid_size, num_channels))
        y = np.zeros((grid_size, grid_size, grid_size, 4))
        
        # these two constants just derived from the
        # filename to locate amino acid name (see above)
        s = len(folder) + 12
        e = s + 3
        
        for sample_idx in range(len(files)):
            label = np.zeros((20), dtype = np.float32)
            label[THE20[files[sample_idx][s:e]]] = 1
            x, y = input_reader(files[sample_idx])
            yield x, y, label
    
    def __new__(cls, batch_size:int = 32, num_channels:int = 27,\
                     grid_size:int = GRID_SIZE, randomize:bool = True,\
                     remove_sidechains:str = 'all', folder:str = ''):
        ds = tf.data.Dataset.range(2)
        ds = ds.interleave(lambda x: tf.data.Dataset.from_generator(
                                     cls._generator,
                                     output_signature = (tf.TensorSpec(shape = (grid_size, grid_size, grid_size, num_channels), dtype = tf.float32),\
                                                         tf.TensorSpec(shape = (grid_size, grid_size, grid_size, 4), dtype = tf.float32),\
                                                         tf.TensorSpec(shape = (20), dtype = tf.float32)),
                                     args = (num_channels, grid_size, randomize, folder, remove_sidechains)),
                           cycle_length = 2,
                           block_length = 1,
                           num_parallel_calls = 2)
        return ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)