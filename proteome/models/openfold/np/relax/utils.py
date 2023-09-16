# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils for minimization."""
import numpy as np

from proteome.constants import residue_constants


def assert_equal_nonterminal_atom_types(
    atom_mask: np.ndarray, ref_atom_mask: np.ndarray
):
    """Checks that pre- and post-minimized proteins have same atom set."""
    # Ignore any terminal OXT atoms which may have been added by minimization.
    oxt = residue_constants.atom_order["OXT"]
    no_oxt_mask = np.ones(shape=atom_mask.shape, dtype=np.bool)
    no_oxt_mask[..., oxt] = False
    np.testing.assert_almost_equal(ref_atom_mask[no_oxt_mask], atom_mask[no_oxt_mask])
