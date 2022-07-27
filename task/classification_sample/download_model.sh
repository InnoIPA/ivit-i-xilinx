# Download Model

# Defince
URL="https://www.xilinx.com/bin/public/openDownload?filename=resnet50_tf2-zcu102_zcu104_kv260-r1.4.0.tar.gz"
FILE="resnet50_tf2-zcu102_zcu104_kv260-r1.4.0.tar.gz"
FOLD="`dirname ${0}`/model"

# Download from Model Zoo
wget ${URL} -O ${FILE}

# Create model and extract model into it
mkdir ${FOLD}
tar -zxvf ${FILE} -C ${FOLD} --strip-components=1
