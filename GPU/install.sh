sudo apt-get update
sudo apt-get install python3-pip python3-h5py python3-pillow
wget https://nvidia.box.com/shared/static/veo87trfaawj5pfwuqvhl6mzc5b55fbj.whl -O torch-1.1.0a0+b457266-cp36-cp36m-linux_aarch64.whl
pip3 install numpy torch-1.1.0a0+b457266-cp36-cp36m-linux_aarch64.whl
sudo pip3 install torchvision
pip3 install bitstring
pip3 install pycuda
rm torch-1.1.0a0+b457266-cp36-cp36m-linux_aarch64.whl
python3 prepare.py
