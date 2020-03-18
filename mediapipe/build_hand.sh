#! /bin/sh
#
# build_hand.sh
# Copyright (C) 2020 weihao <weihao@weihao-G7>
#
# Distributed under terms of the MIT license.
#


bazel-1.2.1 build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
    mediapipe/examples/desktop/hand_tracking:hand_tracking_cpu
