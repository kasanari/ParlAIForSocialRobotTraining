#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Download and build the data if it does not exist.

from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data
import os
import shutil

SOURCE = "/home/jakob/process-neil-data/"

FILES = [
    "train.csv",
    "valid.csv"
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'neil')
    version = '1.0'

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for file_to_copy in FILES:
            shutil.copy(os.path.join(SOURCE, file_to_copy), dpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
