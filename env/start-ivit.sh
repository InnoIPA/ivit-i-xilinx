#!/bin/bash

if [[ ! "${0}" = '/bin/sh' ]]; then
    REST='\e[0m'
    RED='\e[0;31m'
    BRED='\e[7;31m'
    echo -e """ \
    \r${BRED}[WARNING]${REST} \
    Please run \"${RED}source ./env/start-ivit.sh${REST}\" \
    to make sure variable will be set in environment """
    exit
fi

# Store the utilities
FILE=$(realpath $BASH_SOURCE)
ROOT=$(dirname "${FILE}")
source "${ROOT}/utils.sh"

# Move to Correct Path
printd "Initialize iVIT-I-Xilinx ..." G
cd ${ROOT} | exit

# Load APP
./env/load-xilinx-app.sh VCU200EDDPU200B4096

# Export iVIT
if [[ -z $(echo ${IVIT_DEBUG}) ]];then
    echo 'export IVIT_DEBUG=True' >> ~/.bashrc
    source ~/.bashrc
fi

# Verify
if [[ -z $(echo ${IVIT_DEBUG}) ]]; then FLAG="FAILED"; COLOR="BR" ; else FLAG="PASS"; COLOR="BG" ; fi
printd "Verifying iVIT-I ... ${FLAG}" ${COLOR}