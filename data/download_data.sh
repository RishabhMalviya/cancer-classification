sudo apt-get update && sudo apt-get install -y unzip
wget "https://zenodo.org/record/1214456/files/NCT-CRC-HE-100K.zip?download=1" -O NCT-CRC-HE-100K.zip
unzip NCT-CRC-HE-100K.zip
rm ./NCT-CRC-HE-100K.zip
mkdir ./splits
