#! /bin/sh
#
# hand.sh
# Copyright (C) 2020 weihao <weihao@weihao-G7>
#
# Distributed under terms of the MIT license.
#

GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_cpu \
    --calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt
