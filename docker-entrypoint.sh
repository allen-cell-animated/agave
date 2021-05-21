#!/bin/bash

# Xvfb :99 -ac -screen 0 "$XVFB_RES" -nolisten tcp $XVFB_ARGS &
# XVFB_PROC=$!
# sleep 1
# export DISPLAY=:99

cd /agave/build
# agave -platform offscreen --server

file="$(agave -platform offscreen --server)" && echo $file
echo $(agave -platform offscreen --server)
# kill $XVFB_PROC