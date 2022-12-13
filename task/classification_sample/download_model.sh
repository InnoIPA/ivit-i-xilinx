#!/bin/bash

# ******************************************************************************
cat << EOF > /dev/null

iVIT-I Download Script

* Feature:
	1. Provide gdown and wget to download file.
	2. Parsing URL to choose the way to download file. 
* How to use:
	1. Define Parameters and make sure the file name is correct. 
EOF

# Parameters
FOLDER="mobilenet_1_0_224_tf2"
MODEL="mobilenet_1_0_224_tf2.xmodel"

MODEL_URL="https://www.xilinx.com/bin/public/openDownload?filename=mobilenet_1_0_224_tf2-zcu102_zcu104_kv260-r1.4.1.tar.gz"
MODEL_FILE="mobilenet_1_0_224_tf2-zcu102_zcu104_kv260-r1.4.1.tar.gz"

LABEL_URL="https://drive.google.com/file/d/13nmiw_RbZ_pHVUOh9pnD0razxhWhKV6C/view?usp=sharing"
LABEL_FILE="imagenet.txt"

# ******************************************************************************

function down_by_wget(){
	URL=$1; FILE=$2; FOLDER=$3;
    wget "${URL}" -O "${FILE}"
    tar -zxvf "${FILE}" -C "${FOLDER}" --strip-components=1 && rm "${FILE}"
}

function down_by_gdown(){
	URL=$1; FILE=$2; FOLDER=$3;
	gdown --fuzzy "${URL}" -O "${FILE}" && mv "${FILE}" "${FOLDER}"
}

function use_gdown(){
	if [[ $1 == *"google"* ]]; then echo 'true'; else echo 'false'; fi
}

function down(){
	URL=$1; FILE=$2; FOLDER=$3;
	if [[ $(use_gdown "${URL}" ) = 'true' ]]; then
		down_by_gdown "${URL}" "${FILE}" "${FOLDER}"
	else
		down_by_wget "${URL}" "${FILE}" "${FOLDER}"
	fi
}

# Default
ROOT=$(dirname "$(dirname "$(dirname "${0}")")") # ./task/cls/script, 3 levels
MODEL_ROOT="model"

# Move to the target place
cd "${ROOT}" || exist

# Concatenate Path
TRG_ROOT="${MODEL_ROOT}/${FOLDER}"
MODEL_PATH="${TRG_ROOT}/${MODEL}"
LABEL_PATH="${TRG_ROOT}/${LABEL_FILE}"

cat << EOF

# Download Information

MODEL_URL	: ${MODEL_URL}
MODEL_PATH	: ${MODEL_PATH}
LABEL_URL	: ${LABEL_URL}
LABEL_PATH	: ${LABEL_PATH}
EOF

# Create model if need
if [[ ! -d $TRG_ROOT ]];then mkdir -p "${TRG_ROOT}"; fi

# Download from Model Zoo
printf "\n# Download Model \n"
if [[ -f "${MODEL_PATH}"  ]]; then 
	echo "$(date +"%F %T") Model already exist";
else
    echo "$(date +"%F %T") Downloading Model ..."    
	down "${MODEL_URL}" "${MODEL_FILE}" "${TRG_ROOT}"
fi;

# Download from Google Drive
printf "\n# Download Label \n"
if [[ -f "${LABEL_PATH}" ]]; then
	echo "$(date +"%F %T") Label already exist"
else
    echo "$(date +"%F %T") Downloading Label ..."
	down "${LABEL_URL}" "${LABEL_FILE}" "${TRG_ROOT}"
fi