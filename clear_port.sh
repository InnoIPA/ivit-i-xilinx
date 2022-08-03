PORT=$1

if [[ -z `node -v` ]];then
    sudo dnf install npm -y
fi

if [[ -z $PORT ]];then
    echo "Use default port: 820"
    PORT="820"
fi

npx kill-port $PORT