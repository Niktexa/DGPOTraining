#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2021 The TARTRL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""""""

import gymnasium
from .pong_wrappers import make_atari, wrap_deepmind

def make_env(env_id, seed=0, rank=0, log_dir=None, allow_early_resets=True, add_monitor=True, frame_stack=4):
    def _thunk():
        env = make_atari(env_id)  # env_id will be "PongNoFrameskip-v4"
        env = wrap_deepmind(env, frame_stack=True, pytorch_img=True, vector_input=True, multi_agent=True)
        return env
    return _thunk