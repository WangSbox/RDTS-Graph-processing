# -*- coding: utf-8 -*-
from __future__ import print_function
import os

os.system("pip install --upgrade pip")
os.system("pip install torch==1.4.0")
os.system("pip install torchvision==0.5.0")
os.system("pip uninstall torch -y")
os.system("pip uninstall torchvision -y")
os.system("pip install s3fs")
os.system("pip install pandas==1.0.4")
os.system("pip install h5py==3.1.0")
os.system("pip install numpy==1.14.5")
os.system("pip install tqdm==4.50.0")
os.system("pip install cmake==3.21.2")

# os.system("pip install ../../torch-1.8.0+cu101-cp36-cp36m-linux_x86_64.whl")
# os.system("pip install ../../torch-1.8.0+cu101-cp36-cp36m-linux_x86_64.whl")
# os.system("pip install ../../torchvision-0.9.0+cu101-cp36-cp36m-linux_x86_64.whl")
# os.system("pip install ../../torchtext-0.9.0-cp36-cp36m-linux_x86_64.whl")
# os.system("pip install ../../torch_scatter-2.0.6-cp36-cp36m-linux_x86_64.whl")
# os.system("pip install ../../torch_sparse-0.6.9-cp36-cp36m-linux_x86_64.whl")

os.system("pip install torch_geometric")
print('begin importing torch_geomrtoric pkg')
print('pkg has been imported')

os.system("python ./graphdetector/graphdetector.py --epoch 50 --bs 2048 --nw 8 --lr 1e-3 --lc 20 --tgac 0.85")
