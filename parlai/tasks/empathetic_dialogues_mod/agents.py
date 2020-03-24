#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
from parlai.core.teachers import FixedDialogTeacher
from parlai.tasks.empathetic_dialogues.agents import EmpatheticDialoguesTeacher
from .build import build
import numpy as np


DEFAULT_TRAIN_EXPERIENCER_ONLY = False


class EmpatheticDialoguesExtraTeacher(EmpatheticDialoguesTeacher):
    
    def get(self, episode_idx, entry_idx=0):
        ep = self.data[episode_idx]
        ep_i = ep[entry_idx]
        episode_done = entry_idx >= (len(ep) - 1)

        distractor_ep = self.data[random.randrange(self.num_episodes())]
        distractor_ep_i = distractor_ep[0]
        
        action = {
            'situation': ep_i[3],
            'emotion': ep_i[2],
            'text': ep_i[0],
            'labels': [ep_i[1]],
            'distractor_label': [distractor_ep_i[1]],
            'prepend_ctx': ep_i[6],
            'prepend_cand': ep_i[7],
            'deepmoji_ctx': ep_i[4],
            'deepmoji_cand': ep_i[5],
            'episode_done': episode_done,
            'label_candidates': [ep_i[1], distractor_ep_i[1]],
        }
        return action

class DefaultTeacher(EmpatheticDialoguesExtraTeacher):
    pass
