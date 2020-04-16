#!/usr/bin/env python3

import os
from typing import Any, List
import csv
import random

import numpy as np

from parlai.core.teachers import DialogTeacher, FixedDialogTeacher
from .build import build


DEFAULT_TRAIN_EXPERIENCER_ONLY = False
class NeilTeacher(FixedDialogTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.opt = opt
        base_datatype = self.datatype.split(':')[0]
        self.datapath = os.path.join(
            self.opt['datapath'],
            'neil',
            base_datatype + '.csv',
        )
        self.experiencer_side_only = (
            opt.get('train_experiencer_only', DEFAULT_TRAIN_EXPERIENCER_ONLY)
            and base_datatype == 'train'
        ) or base_datatype != 'train'
        print(
            f'[NeilTeacher] Only use experiencer side? '
            f'{self.experiencer_side_only}, datatype: {self.datatype}'
        )

        build(opt)
        self._setup_data(base_datatype)

        self.num_exs = sum([len(d) for d in self.data])
        self.num_eps = len(self.data)
        self.reset()

    @classmethod
    def add_cmdline_args(cls, argparser):
        agent = argparser.add_argument_group('Neil teacher arguments')
        agent.add_argument(
            '--train-experiencer-only',
            type='bool',
            default=DEFAULT_TRAIN_EXPERIENCER_ONLY,
            # i.e. do not include the other side of the conversation where the Listener
            # (responder) utterance would be the text and the Speaker (experiencer)
            # utterance would be the label
            help='In the train set, only use Speaker (experiencer) utterances as text and Listener (responder) utterances as labels.',
        )
    def num_episodes(self):
        return self.num_eps

    def num_examples(self):
        return self.num_exs

    def _setup_data(self, base_datatype):

        with open(self.datapath) as csvfile:
            reader = csv.DictReader(csvfile)
            self.data = []
            for episode_index, row in enumerate(reader):
                episode = []

                scene_with_direction = " ".join((row["scene"], row['direction'])) # Description of scene combined with "how would the robot responsd ... ?"

                if row['robot_line'] == '' and row['human_line'] == '': 
                    # Robot makes comment based on situation, 
                    # input: scene + direction 
                    # label: utterance
                    episode.append([scene_with_direction, row["utterance"], row["dominant_affect"], row["scene"]])

                elif row['robot_line'] == '' and row['human_line'] != '': 
                    # Human says something to robot
                    # input: scene + direction
                    # label: utterance
                    episode.append([scene_with_direction, row["utterance"], row["dominant_affect"], row["scene"]])

                elif row['robot_line'] != '' and row['human_line'] == '': 
                    # Robot asks something to human
                    # input: scene + direction
                    # label: utterance
                    episode.append([scene_with_direction, row["utterance"], row["dominant_affect"], row["scene"]])

                else: # Conversation between human and robot
                    episode.append([row["scene"], row["robot_line"], row["dominant_affect"], row["scene"]])
                    episode.append([row["human_line"], row["utterance"], row["dominant_affect"], row["scene"]])

                self.data += [episode]

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
            'episode_done': episode_done,
            'label_candidates': [ep_i[1], distractor_ep_i[1]],
        }
        return action

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        return shared


class DefaultTeacher(NeilTeacher):
    pass
