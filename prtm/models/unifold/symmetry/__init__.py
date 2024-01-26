# Copyright 2022 DeepMind Technologies Limited
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
""" Implementation for UF-Symmetry """
from prtm.models.unifold.symmetry.assemble import assembly_from_prediction
from prtm.models.unifold.symmetry.config import uf_symmetry_config
from prtm.models.unifold.symmetry.dataset import load_and_process_symmetry
from prtm.models.unifold.symmetry.model import UFSymmetry
