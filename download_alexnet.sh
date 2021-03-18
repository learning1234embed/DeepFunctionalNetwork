#!/bin/bash

echo "Downloading AlexNet..."

if [ ! -d "obstacle" ]; then
	mkdir obstacle
fi

fileid="1jF5O5r89bKHuaEAU43VnbNqMqlNgfDeG"
filename="obstacle/alexnet.pb"
echo "Downloading the model... ${filename}"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

fileid="1ze8L8t7GU8bcZjhCTjp-ePhDzY4MTmYk"
filename="obstacle/execute_alexnet.py"
echo "Downloading the execution code... ${filename}"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
