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

import numpy as np
from .envs import make_env

class PongEnv:  # CHANGE: Class name
    def __init__(self, args):
        env = make_env(args.game_name, seed=0, rank=0,
                      log_dir=None, allow_early_resets=True, add_monitor=True, frame_stack=4)()
        # CHANGE: Remove ToMultiAgent call (already done in wrap_deepmind)
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.share_observation_space = env.observation_space
    
    def step(self, a):
        a = a[0]  # Extract single action from multi-agent format
        return self.env.step(a)
    
    def reset(self):
        return self.env.reset()
    
    def close(self):
        self.env.close()
    
    def seed(self, s):
        if type(s) == type(np.array([1])[0]):
            self.env.seed(s.item())
        else:
            self.env.seed(s)
    
    def render(self, mode='human'):
        img = self.env.render(mode)
        return img