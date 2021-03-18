#!/bin/bash

echo "Downloading the obstacle dataset..."

if [ ! -d "obstacle" ]; then
	mkdir obstacle
fi

fileid="1vLbdE4XgvlCKYjV_gzV5wGwp9kZfxogs"
filename="obstacle/obstacle_test_data.npy"
echo "Downloading the test data... ${filename}"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

fileid="1-hP0QYiwtC1Fta5djImZnQXnoPbJcz1-"
filename="obstacle/obstacle_test_label.npy"
echo "Downloading the test label... ${filename}"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

fileid="1reAfUUamzFvxd2cGdoGm9-7XDpTR_MC7"
filename="obstacle/obstacle_train_data.npy"
echo "Downloading the train data... ${filename}"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

fileid="1gNMuJRaydw_GofjgHW8GX9HkWVdgAgHh"
filename="obstacle/obstacle_train_label.npy"
echo "Downloading the train label... ${filename}"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
