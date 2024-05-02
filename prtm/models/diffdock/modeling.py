import copy
import os
from argparse import ArgumentParser, FileType, Namespace
from dataclasses import asdict
from functools import partial

import nglview as nv
import numpy as np
import pandas as pd
import py3Dmol
import torch
import yaml
from rdkit import Chem, RDLogger
from rdkit.Chem import RemoveAllHs
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from prtm.models.diffdock.utils.diffusion_utils import get_t_schedule
from prtm.models.diffdock.utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
from prtm.models.diffdock.utils.inference_utils import InferenceDataset
from prtm.models.diffdock.utils.sampling import randomize_position, sampling
from prtm.models.diffdock.utils.utils import get_model
from prtm.models.diffdock.utils.visualise import PDBBlockFile
from prtm.models.esm.config import ESM2Config
from prtm.models.esm.data import FastaBatchedDataset
from prtm.models.esm.esm2 import ESM2
from prtm.models.esm.modeling import ESMForSequenceEmbedding
from prtm.protein import Protein14, Protein37
