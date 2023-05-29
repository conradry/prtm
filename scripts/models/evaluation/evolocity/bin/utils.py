import datetime
import errno
import math
import os
import random
import sys
import time
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats as ss
import seaborn as sns
from anndata import AnnData
from Bio import BiopythonWarning
from dateutil.parser import parse as dparse
from fuzzywuzzy import fuzz
from fuzzywuzzy import process as fuzzyproc
from scipy.sparse import csr_matrix, dok_matrix

warnings.simplefilter("ignore", BiopythonWarning)
from Bio import Seq, SeqIO

np.random.seed(1)
random.seed(1)


def tprint(string):
    string = str(string)
    sys.stdout.write(str(datetime.datetime.now()) + " | ")
    sys.stdout.write(string + "\n")
    sys.stdout.flush()


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def iterate_lengths(lengths, seq_len):
    curr_idx = 0
    for length in lengths:
        if length > seq_len:
            sys.stderr.write(
                "Warning: length {} greather than expected "
                "max length {}\n".format(length, seq_len)
            )
        yield (curr_idx, curr_idx + length)
        curr_idx += length
