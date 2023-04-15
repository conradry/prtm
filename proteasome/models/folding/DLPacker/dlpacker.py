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

import numpy as np

from Bio.PDB import PDBParser, Selection, Superimposer, PDBIO, Atom, Residue, Structure
from utils import DLPModel, InputBoxReader, DataGenerator, THE20, SCH_ATOMS, BB_ATOMS, SIDE_CHAINS, BOX_SIZE

class DLPacker():
    # This is the meat of our code.
    # This class reads in PDB files using biopython
    # and implements various operations on them
    # using DLPModel's predictions.
    # You might need to change this class if you
    # want to implement some new functionality
    def __init__(self, str_pdb:str, model:DLPModel = None,\
                       input_reader:InputBoxReader = None):
        # Input:
        # str_pdb      - filename of the PDB structure we will be working with
        #                this structure might or might not contain the side
        #                chains it does not really matter, as they will always
        #                be removed before repacking
        # model        - DLPModel instance
        # input_reader - InputBoxReader instance
        self.parser = PDBParser(PERMISSIVE = 1) # PDB files reader
        self.sup = Superimposer() # superimposer
        self.io = PDBIO() # biopython's IO lib
        
        self.box_size = BOX_SIZE  # do not change
        self.altloc = ['A', 'B']  # initial altloc selection order preference
        
        self.str_pdb = str_pdb
        self.ref_pdb = './reference.pdb' # reference atoms to align residues to
        self._read_structures()
        self.reconstructed = None
        
        self.lib_name = './library.npz' # library of rotamers
        self._load_library()
        
        self.model = model
        if not self.model:
            self.model = DLPModel(width = 128, nres = 6)
            self.model.load_model(weights = 'DLPacker_weights')

        self.input_reader = input_reader
        if not self.input_reader: self.input_reader = InputBoxReader()
    
    def _load_library(self):
        # Loads library of rotamers.
        # Original library uses float16 to save space, but here we convert
        # everything to float32 to speed up calculations
        self.library = np.load(self.lib_name, allow_pickle = True)
        self.library = self.library['arr_0'].item()
        for k in self.library['grids']:
            self.library['grids'][k] = self.library['grids'][k].astype(np.float32)
    
    def _read_structures(self):
        # Reads in main PDB structure and reference structure.
        self.structure = self.parser.get_structure('structure', self.str_pdb)
        self.reference = self.parser.get_structure('reference', self.ref_pdb)
        
        self._remove_hydrogens(self.structure) # we never use hydrogens
        self._convert_mse(self.structure)      # convers MSE to MET
        self._remove_water(self.structure)     # waters are not used anyway
        self._remove_altloc(self.structure)    # only leave one altloc
    
    def _remove_hydrogens(self, structure:Structure):
        # Removes all hydrogens.
        # This code is not suited to work with hydrogens
        for residue in Selection.unfold_entities(structure, 'R'):
            remove = []
            for atom in residue:
                if atom.element == 'H': remove.append(atom.get_id())
            for i in remove: residue.detach_child(i)
    
    def _convert_mse(self, structure:Structure):
        # Changes MSE residues to MET
        for residue in Selection.unfold_entities(structure, 'R'):
            if residue.get_resname() == 'MSE':
                residue.resname = 'MET'
                for atom in residue:
                    if atom.element == 'SE':
                        new_atom = Atom.Atom('SD',\
                                             atom.coord,\
                                             atom.bfactor,\
                                             atom.occupancy,\
                                             atom.altloc,\
                                             'SD  ',\
                                             atom.serial_number,\
                                             element='S')
                        residue.add(new_atom)
                        atom_to_remove = atom.get_id()
                residue.detach_child(atom_to_remove)
    
    def _remove_water(self, structure:Structure):
        # Removes all water molecules
        residues_to_remove = []
        for residue in Selection.unfold_entities(structure, 'R'):
            if residue.get_resname() == 'HOH':
                residues_to_remove.append(residue)
        for r in residues_to_remove:
            r.get_parent().detach_child(r.get_id())
    
    def _remove_altloc(self, structure:Structure):
        # Only leaves one altloc with the largest sum of occupancies
        total_occupancy = {}
        for atom in Selection.unfold_entities(structure, 'A'):
            if atom.is_disordered():
                for alt_atom in atom:
                    occupancy = alt_atom.get_occupancy()
                    if alt_atom.get_altloc() in total_occupancy:
                        total_occupancy[alt_atom.get_altloc()] += occupancy
                    else:
                        total_occupancy[alt_atom.get_altloc()] = occupancy

        # optionally select B altloc if it has larger occupancy
        # rare occasion, but it happens
        if 'A' in total_occupancy and 'B' in total_occupancy:
            if total_occupancy['B'] > total_occupancy['A']:
                self.altloc = ['B', 'A']
        
        # only leave one altloc
        disordered_list, selected_list = [], []
        for residue in Selection.unfold_entities(structure, 'R'):
            for atom in residue:
                if atom.is_disordered():
                    disordered_list.append(atom)
                    # sometimes one of the altlocs just does not exist!
                    try:
                        selected_list.append(atom.disordered_get(self.altloc[0]))
                    except:
                        selected_list.append(atom.disordered_get(self.altloc[1]))
                    selected_list[-1].set_altloc(' ')
                    selected_list[-1].disordered_flag = 0
        
        for d, a in zip(disordered_list, selected_list):
            p = d.get_parent()
            p.detach_child(d.get_id())
            p.add(a)
    
    def _get_residue_tuple(self, residue:Residue):
        # Returns (number, chain, name) tuple for given residue
        # Example: (10, 'B', 'PHE')
        r = residue.get_id()[1]
        s = residue.get_full_id()[2]
        n = residue.get_resname()
        return (r, s, n)
    
    def _get_parent_structure(self, residue:Residue):
        # Returns the parent structure of the given residue
        return residue.get_parent().get_parent().get_parent()
    
    def _align_residue(self, residue:Residue):
        # In order to generate input box properly
        # we first need to align selected residue
        # to reference atoms from reference.pdb
        if not residue.has_id('N') or not residue.has_id('C') or not residue.has_id('CA'):
            print('Missing backbone atoms: residue', self._get_residue_tuple(residue))
            return False
        r = list(self.reference.get_atoms())
        s = [residue['N'], residue['CA'], residue['C']]
        self.sup.set_atoms(r, s)
        self.sup.apply(self._get_parent_structure(residue))
        return True
    
    def _align_structures(self, structure_a:Structure, structure_b:Structure):
        # Aligns two structures using backbone atoms
        bb_a, bb_b = [], []
        residues_a = Selection.unfold_entities(structure_a, 'R')
        residues_b = Selection.unfold_entities(structure_b, 'R')
        for a, b in zip(residues_a, residues_b):
            for n in BB_ATOMS:
                if a.has_id(n):
                    bb_a.append(a[n])
                    bb_b.append(b[n])
                    
        self.sup.set_atoms(bb_a, bb_b)
        self.sup.apply(structure_b)
    
    def _get_box_atoms(self, residue:Residue):
        # Alighns selected residue to reference positions
        # and selects atoms that lie withing some cube.
        # Cube size is 10 angstroms by default and an
        # additional offset of 1 angstrom is used
        # to include all atoms bc even if the atom is
        # slightly outside the cube, due to gaussian kernel
        # blur, some of the density might still be within
        # 10 angstrom
        aligned = self._align_residue(residue)
        if not aligned: return []
        atoms = []
        b = self.box_size + 1 # one angstrom offset to include more atoms
        for a in self._get_parent_structure(residue).get_atoms():
            xyz = a.coord
            if xyz[0] < b and xyz[0] > -b and\
               xyz[1] < b and xyz[1] > -b and\
               xyz[2] < b and xyz[2] > -b:
                atoms.append(a)
        return atoms
    
    def _genetare_input_box(self, residue:Residue, allow_missing_atoms:bool = False):
        # Takes a residue and generates a special
        # dictionary that is then given to InputReader,
        # which uses this dictionary to generate the actual input
        # for the neural network
        # Input:
        # residue             - the residue we want to restore
        # allow_missing_atoms - boolean flag that allows or disallows
        #                       missing sidechain atoms
        atoms = self._get_box_atoms(residue)
        if not atoms: return None
        
        r, s, n = self._get_residue_tuple(residue)
        
        exclude, types, resnames = [], [], []
        segids, positions, names = [], [], []
        resids = []
        
        for i, a in enumerate(atoms):
            p = a.get_parent()
            a_tuple = (p.get_id()[1], p.get_full_id()[2], p.get_resname())
            if a.get_name() not in BB_ATOMS and (r, s, n) == a_tuple:
                exclude.append(i)
            
            types.append(a.element)
            resnames.append(a.get_parent().get_resname())
            segids.append(a.get_parent().get_full_id()[2])
            positions.append(a.coord)
            names.append(a.get_name())
            resids.append(a.get_parent().get_id()[1])
            
        d = {'target': {'id': int(r), 'segid': s, 'name': n, 'atomids': exclude},\
             'types': np.array(types),\
             'resnames': np.array(resnames),\
             'segids': np.array(segids),\
             'positions': np.array(positions, dtype = np.float16),\
             'names': np.array(names),\
             'resids': np.array(resids)}
        
        if allow_missing_atoms or len(exclude) == SCH_ATOMS[n]:
            return d
        else:
            return None
    
    def _get_sorted_residues(self, structure:Structure, targets:list = [],\
                                   method:str = 'sequence'):
        # Sorts the residues in the structure according to
        # one of the heuristics that we describe in the paper
        # Input:
        # structure - structure we want to restore sidechains in
        # targets   - optional list of residues we are interested in
        #             all the residues in the structure are used if
        #             this list is empty
        # method    - method to use to sort residues as destribed in
        #             the paper:
        #                sequence - just from N- to C- terminus as
        #                           read by biopython
        #                natoms   - number of atoms around residue
        #                score    - NN's prediction quality
        assert method in ['natoms', 'score', 'sequence'],\
              'Method should be natoms, sequence or score!'
        
        if method == 'sequence':
            out = []
            for residue in Selection.unfold_entities(structure, 'R'):
                if not targets or self._get_residue_tuple(residue) in targets:
                    out.append(residue)
            return out
        
        elif method == 'natoms':
            print('Sorting residues...')
            tuples = []
            for residue in Selection.unfold_entities(structure, 'R'):
                if not targets or self._get_residue_tuple(residue) in targets:
                    if residue.get_resname() in THE20:
                        if residue.has_id('CA') and residue.has_id('C') and residue.has_id('N'):
                            atoms = self._get_box_atoms(residue)
                            tuples.append((residue, len(atoms)))
            tuples.sort(key = lambda x: -x[1])
        
        elif method == 'score':
            tuples = []
            for i, residue in enumerate(Selection.unfold_entities(structure, 'R')):
                if not targets or self._get_residue_tuple(residue) in targets:
                    if residue.get_resname() in THE20 and residue.get_resname() != 'GLY':
                        name = self._get_residue_tuple(residue)
                        print("Scoring residue:", i, name, end = '\r')

                        r, s, n = self._get_residue_tuple(residue)
                        box = self._genetare_input_box(residue, True)

                        if not box:
                            print("\nSkipping residue:", i, residue.get_resname())
                            continue

                        pred = self._get_prediction(box, n)
                        scores = np.abs(self.library['grids'][n] - pred)
                        scores = np.mean(scores, axis = tuple(range(1, pred.ndim+1)))
                        best_ind = np.argmin(scores)
                        best_score = np.min(scores)
                        tuples.append((residue, best_score / SCH_ATOMS[n]))
            tuples.sort(key = lambda x: x[1])
        
        return [r for r, c in tuples]
    
    def _remove_sidechains(self, structure:Structure):
        # Removes all sidechains from the given structure
        for residue in Selection.unfold_entities(structure, 'R'):
            if residue.get_resname() in THE20:
                self._remove_sidechain(residue)
    
    def _remove_sidechain(self, residue:Residue):
        # Removes sidechain from the given residue
        l = []
        for atom in residue:
            if atom.get_name() not in BB_ATOMS:
                l.append(atom.get_id())
        for d in l: residue.detach_child(d)
    
    def _get_prediction(self, box:dict, label:str):
        # Runs NN prediction to get density

        # prepare input
        i, _ = self.input_reader(box)
        labels = np.zeros((1, 20), dtype = np.float32)
        labels[0, THE20[label]] = 1.0
        
        # run the model
        o = self.model.model([i[None, ...], labels]).numpy()[0]
        
        # subtract the input to only leave target side chain
        pred = o - i[..., :4]
        pred[pred < 0] = 0
        
        # At this point we have a prediction from the NN.
        # The following code implements a couple of optimizations
        # to speed up the reconstruction process:
        # 1. Truncate the box to exclude outer voxels which
        #    never contain target side chain atoms.
        # 2. Downsample the box to speed up the process.
        #    This reduces the quality slightly, but saves a lot
        #    of time and memory.
        # 3. Despite that we predict 4 element channels separately,
        #    for most amino acids we can sum all four of them into
        #    one without any loss of information. The only exceptions
        #    from this rule are His, Gln and Asp, so we will treat
        #    them differently.
        
        # truncate
        pred = pred[5:-5, 5:-5, 5:-5, :]
        
        # downsample
        dpred = np.zeros((15, 15, 15, 4))
        for i in range(0, 30, 2):
            for j in range(0, 30, 2):
                for k in range(0, 30, 2):
                    v = np.mean(pred[i:i+2, j:j+2, k:k+2, :], axis = (0, 1, 2))
                    dpred[i//2, j//2, k//2, :] = v
        
        # summ all the channels if not Asn, Gln or His
        if label not in ['ASN', 'GLN', 'HIS']: dpred = np.sum(dpred, axis = -1)
                
        return dpred
    
    def mutate_sequence(self, target:tuple, new_label:str):
        # Takes as input residue tuple [like (10, 'B', 'PHE')]
        # and mutates it in the sequence to new one given by new_label argument
        # IMPORTANT: this function just renames a residue without
        # doing anything else at all
        assert new_label in THE20,\
              'Only mutations to canonical 20 amino acids are supported!'
        for residue in Selection.unfold_entities(self.structure, 'R'):
            if target == self._get_residue_tuple(residue):
                residue.resname = new_label
                print('Mutated', target, 'to', new_label + '!')
                return residue
        print('Target residue', target, 'was not found!')
        return None
    
    def mutate_residue(self, target:tuple, new_label:str):
        # this function takes as input residue tuple [like (10, 'B', 'PHE')]
        # and mutates it and then runs reconstruction to build new sidechain
        residue = self.mutate_sequence(target, new_label)
        if residue: self.reconstruct_residue(residue, False)
    
    def reconstruct_residue(self, residue:Residue, refine_only:bool = False):
        # Reconstructs side chain for one residue
        # refine_only flag marks that current side chain is
        # already there and we only want to update atom
        # positions instead of building it from scratch
        r, s, n = self._get_residue_tuple(residue)
        box = self._genetare_input_box(residue, True)
        
        if not box:
            print("Skipping residue:", (r, s, n), end = '\n')
            return
        
        pred = self._get_prediction(box, n)
        # this block runs reconstruction of the residue
        scores = np.abs(self.library['grids'][n] - pred)
        scores = np.mean(scores, axis = tuple(range(1, pred.ndim + 1)))
        best_ind = np.argmin(scores)
        best_score = np.min(scores)
        best_match = self.library['coords'][n][best_ind]
        
        if not refine_only: self._remove_sidechain(residue)

        for i, name in enumerate(SIDE_CHAINS[n]):
            if refine_only:
                residue[name].coord = best_match[i]
            else:
                # most values are dummy here
                new_atom = Atom.Atom(name,\
                                     best_match[i],\
                                     0,\
                                     1,\
                                     ' ',\
                                     name,\
                                     2,\
                                     element = name[:1])
                residue.add(new_atom)
    
    def reconstruct_protein(self, refine_only:bool = False,\
                            order:str = 'natoms', output_filename:str = ''):
        # Runs reconstruction process for the whole structure saved as 
        # self.structure and saves the result as self.reconstructed
        assert order in ['natoms', 'score', 'sequence'],\
              'Order should be natoms, sequence or score!'
        
        # create copy of the currents structure as self.reconstructed
        # and remove side chains from it.
        # 
        # If refine_only flag is True, then we don't remove side chains
        # this could be useful to refine existing positions of side chains,
        # however in our experiments this procedure did not significantly
        # improve the performance
        if not self.reconstructed: self.reconstructed = self.structure.copy()
        else: print('Reconstructed structure already exists, something might be wrong!')
        if not refine_only: self._remove_sidechains(self.reconstructed)
        
        # run reconstruction for all residues in selected order
        sorted_residues = self._get_sorted_residues(self.reconstructed, method = order)
        for i, residue in enumerate(sorted_residues):
            if residue.get_resname() in THE20 and residue.get_resname() != 'GLY':
                name = self._get_residue_tuple(residue)
                print("Working on residue:", i, name, end = '\r')
                self.reconstruct_residue(residue, refine_only)
        
        # align reconstructed structure to original one
        self._align_structures(self.structure, self.reconstructed)
        
        # write self.reconstructed is filename is specified
        if output_filename:
            print('\nWriting output file...')
            self.io.set_structure(self.reconstructed)
            self.io.save(output_filename)
        print('Done!')

    def reconstruct_region(self, targets:list, order:str = 'natoms',\
                                 refine_only:bool = False, output_filename:str = ''):
        # Same as reconstruct_protein, but only packs specified list of residues
        # targets is a list of tuples, for example:
        # [(10, 'B', 'PHE'), ('11', 'C', 'VAL'), ... ]
        assert order in ['natoms', 'score', 'sequence'],\
              'Order should be natoms, sequence or score!'
        
        # create a copy of the current structure
        if not self.reconstructed: self.reconstructed = self.structure.copy()
        else: print('Reconstructed structure already exists, something might be wrong!')
        
        # remove side chains for target amino acids is refine_only is False
        if not refine_only:
            for residue in Selection.unfold_entities(self.structure, 'R'):
                if self._get_residue_tuple(residue) in targets:
                    self._remove_sidechain(residue)
        
        # run reconstruction for specified list of residues
        sorted_residues = self._get_sorted_residues(self.reconstructed, targets, method = order)
        for i, residue in enumerate(sorted_residues):
            if self._get_residue_tuple(residue) in targets:
                if residue.get_resname() in THE20 and residue.get_resname() != 'GLY':
                    name = self._get_residue_tuple(residue)
                    print("Working on residue:", i, name, end = '\r')
                    self.reconstruct_residue(residue, refine_only)
        
        self._align_structures(self.structure, self.reconstructed)
        
        if output_filename:
            print()
            print('Writing output file...')
            self.io.set_structure(self.reconstructed)
            self.io.save(output_filename)
        print('Done!')
        
    def save_structure(self, output_filename:str):
        # Writes self.structure to disk
        self.io.set_structure(self.structure)
        self.io.save(output_filename)
    
    def save_reconstructed(self, output_filename:str):
        # Writes self.reconstructed to disk
        self.io.set_structure(self.reconstructed)
        self.io.save(output_filename)
        
    def get_targets(self, target:tuple, radius:float):
        # This is just a small helper function that can give us a list
        # of amino acids withing some radius around our target residue.
        # This can be used if we have mutated one residue and want to
        # refine side chains within a certain radius around it, see an
        # example in the jupyter notebook from the repo
        center = []
        for residue in Selection.unfold_entities(self.structure, 'R'):
            if self._get_residue_tuple(residue) == target:
                for atom in residue:
                    center.append(atom.coord)
                break
        if not center:
            print('Target residue', target, 'was not found!')
            return []
        center = np.array(center)
        center = np.mean(center, axis = 0)
        out = set()
        for atom in Selection.unfold_entities(self.structure, 'A'):
            coord = np.array(atom.coord)
            dist = np.sqrt(np.sum(np.abs(coord - center) ** 2))
            if dist < radius:
                out.add(self._get_residue_tuple(atom.get_parent()))
        return list(out)