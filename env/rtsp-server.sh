#!/bin/bash

function get_pid(){
    ps aux | pgrep rtsp-simple
}

function run_rtsp_server(){
    if [[ ! -z $(get_pid) ]]; then
        printd "RTSP Server already exist ..." Y;
        return
    fi
    
    if [[ ! -f "rtsp-simple-server" ]]; then
        printd "Could not find rtsp-simple-server launcher ... " R; exit; 
    fi

    ./rtsp-simple-server &>./rtsp-simple-server.log &
    printd "Started RTSP Simple Server" Cy

}

function stop_rtsp_server(){
    echo $(get_pid) | xargs kill -9 &>/dev/null
    printd "Stop RTSP Simple Server" R
}

# Store the utilities
FILE=$(realpath "$0")
ROOT=$(dirname "${FILE}")
ARGS=$1
source "${ROOT}/utils.sh"

# Stop
if [[ $ARGS = 'stop' ]];then
    stop_rtsp_server; exit
fi

# Run
cd "${ROOT}/rtsp-simple-server" || exit
run_rtsp_server;