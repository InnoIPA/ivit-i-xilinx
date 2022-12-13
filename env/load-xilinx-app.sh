#!/bin/bash

function find_xlnx_app(){
    ls /lib/firmware/xilinx/ | grep ${1} &2>/dev/null || echo ""
}

# Store the utilities
FILE=$(realpath "$0")
ROOT=$(dirname "${FILE}")
ARG=$1
source "${ROOT}/utils.sh"

# Define App Name
TRG_APP=VCU200EDDPU200B4096

# Get Application
if [[ -z ${ARG} ]]; then
    printd "Using Default Application: ${TRG_APP}"
else
    TRG_APP=${ARG};
    printd "Updated Application: ${TRG_APP}"
fi

# Check Application
if [[ -z "$(find_xlnx_app ${TRG_APP})" ]]; then
    printd "Couldn't find application, please check /lib/firmware/xilinx " R ;
    exit;
fi

# Unload Default App
xmutil unloadapp
printd "Unload Application"

# Load Target App
xmutil loadapp ${TRG_APP}
printd "Loaded Application"
