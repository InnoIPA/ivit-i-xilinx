#!/bin/bash

# Install pre-requirement
if [[ -z $(which jq) ]];then
    echo "Installing requirements .... "
    sudo dnf install jq -yqq
fi

# Variable
CONF="ivit-i.json"
FLAG=$(ls ${CONF} 2>/dev/null)
if [[ -z $FLAG ]];then
    CONF="${RUN_PWD}/${CONF}"
    FLAG=$(ls ${CONF} 2>/dev/null)
    if [[ -z $FLAG ]];then
        echo "Couldn't find configuration (${CONF})"
        exit
    fi
fi

# Parse information from configuration
PORT=$(cat ${CONF} | jq -r '.PORT')
WORKER=$(cat ${CONF} | jq -r '.WORKER')
THREADING=$(cat ${CONF} | jq -r '.THREADING')
export IVIT_I=`realpath ./ivit-i.json`

# Run
if [[ ! -d "./ivit_i/web" ]];then
    echo "Could not found the web api, make sure the submodule is downloaded."
    exit
fi

# get ip address
ip=$(python3 -c "from ivit_i.web.tools.common import get_address;print(get_address())")
# echo "HOST: ${ip}" | boxes -s 80x5 -a c
echo -ne "\n\nHOST: ${ip}\n\n"


# ---------------------------------------------------------------------------------------

# Run Micro Service
./env/stream-server.sh start

# Run Web API
gunicorn \
-w ${WORKER} \
--threads ${THREADING} \
--bind 0.0.0.0:${PORT} \
ivit_i.web.app:app

./env/stream-server.sh stop