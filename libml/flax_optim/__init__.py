# Vendored from flax 0.5.3 (the last release that shipped ``flax.optim``).
#
# Copyright 2022 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trimmed copy of the deprecated ``flax.optim`` API (from flax 0.5.3).

Only the optimizers used by the L2P training code are included:
Adam, Momentum, GradientDescent and the Optimizer/MultiOptimizer/
ModelParamTraversal machinery. Loaded as ``flax.optim`` by
``libml.flax_optim_compat``.
"""

# pylint: disable=g-multiple-import
from .base import (
    OptimizerState, OptimizerDef, Optimizer, MultiOptimizer,
    ModelParamTraversal)
from .adam import Adam
from .momentum import Momentum
from .sgd import GradientDescent

__all__ = [
    'Adam',
    'OptimizerState',
    'OptimizerDef',
    'Optimizer',
    'MultiOptimizer',
    'ModelParamTraversal',
    'Momentum',
    'GradientDescent',
]
# pylint: enable=g-multiple-import
