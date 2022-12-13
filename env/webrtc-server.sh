#!/bin/bash
function check_webrtc_server(){
	echo "$( docker ps | grep ${1})"
}

function webrtc_server_docker_cli(){
    CNT_NAME=$1
    docker run --rm -d \
    --name ${CNT_NAME} \
    --network host \
    ghcr.io/deepch/rtsptoweb:latest
}

function run_webrtc_server(){
    CNT_NAME=$1

    if [[ -n $(check_webrtc_server "${CNT_NAME}") ]]; then
        printd "WebRTC Server might already exist ..." Y
    else
        webrtc_server_docker_cli ${CNT_NAME};
        printd "Started WebRTC Server" G    
    fi
}

function stop_webrtc_server(){
    CNT_NAME=$1
	docker stop "${CNT_NAME}" &>/dev/null
    printd "Stop WebRTC Server" R
}

# Store the utilities
CNT_NAME="ivit-webrtc-server"
FILE=$(realpath "$0")
ROOT=$(dirname "${FILE}")
ARGS=$1
source "${ROOT}/utils.sh"

# Correct Path
cd "${ROOT}" || exit

# Stop
if [[ ${ARGS} = 'stop' ]]; then
    stop_webrtc_server ${CNT_NAME}; exit
fi

# Run
run_webrtc_server ${CNT_NAME};
