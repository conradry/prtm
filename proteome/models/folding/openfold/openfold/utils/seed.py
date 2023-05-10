import logging
import os
import random

import numpy as np
from proteome.models.folding.openfold.openfold.utils.suppress_output import SuppressLogging
from pytorch_lightning.utilities.seed import seed_everything


def seed_globally(seed=None):
    if "PL_GLOBAL_SEED" not in os.environ:
        if seed is None:
            seed = random.randint(0, np.iinfo(np.uint32).max)
        os.environ["PL_GLOBAL_SEED"] = str(seed)
        logging.info(f'os.environ["PL_GLOBAL_SEED"] set to {seed}')

    # seed_everything is a bit log-happy
    with SuppressLogging(logging.INFO):
        seed_everything(seed=None)
