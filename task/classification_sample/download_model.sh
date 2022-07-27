# Download Model
printf "\n"
printf "# Download Model \n"

# Defince
URL="https://www.xilinx.com/bin/public/openDownload?filename=mobilenet_1_0_224_tf2-zcu102_zcu104_kv260-r1.4.1.tar.gz"
FILE="mobilenet_1_0_224_tf2-zcu102_zcu104_kv260-r1.4.1.tar.gz"
ROOT=`dirname ${0}`
FOLD="model"

# Move to the target place
cd ${ROOT}

# Download from Model Zoo
wget ${URL} -O ${FILE}

# Create model and extract model into it
mkdir ${FOLD}
tar -zxvf ${FILE} -C ${FOLD} --strip-components=1

# Remove ZIP
rm ${FILE}