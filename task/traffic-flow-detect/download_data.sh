#!/bin/bash
printf "\n"
printf "# Download File \n"

# Define Parameters 
URL="https://drive.google.com/file/d/16xK3KBJdWKZWxjzTH-_VMxpYfasEFDtz/view?usp=share_link"
GID="16xK3KBJdWKZWxjzTH-_VMxpYfasEFDtz"
TRG_FOLDER="./data"
FILE_NAME="4-corner-downtown-trim-1280x720.mp4"
LEN=20

# Combine Parameter
FILE_PATH="${TRG_FOLDER}/${FILE_NAME}"

# Show information
printf "%-${LEN}s | %-${LEN}s \n" "TRG_FOLDER" "${TRG_FOLDER}"
printf "%-${LEN}s | %-${LEN}s \n" "FILE_NAME" "${FILE_NAME}"
printf "%-${LEN}s | %-${LEN}s \n" "DOWNLOAD_URL" "${URL}"


# Check if folder exist
if [[ ! -d "${TRG_FOLDER}" ]];then
	printf "Create ${TRG_FOLDER} ... "
	mkdir ${TRG_FOLDER}
	if [[ $? == 0 ]];then printf "Done \n";else printf "Failed \n"; fi
fi

# Check if file exist
if [[ ! -f "${FILE_PATH}" ]];then
	printf "Download the file (${FILE_PATH}) ... "
	gdown --id $GID -O ${FILE_PATH} > /dev/null 2>&1
	if [[ $? == 0 ]];then printf "Done \n";else printf "Failed \n"; fi
else
	printf "File alread exist ! \n"
fi