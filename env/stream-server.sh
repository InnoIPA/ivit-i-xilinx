#!/bin/bash

# Get Correct Path
FILE=$(realpath "$0")
ROOT=$(dirname "${FILE}")
ARGS=$1

# Move to Correct Path
cd "${ROOT}" || exit

# RTSP SIMPLE SERVER
./rtsp-server.sh ${ARGS}

# WEBRTC SERVER
./webrtc-server.sh ${ARGS}