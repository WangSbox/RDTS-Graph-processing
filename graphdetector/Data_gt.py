# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import h5py
import torch

def get_data(dtfolder, mode, located=3):
    flist = []
    data_reference = os.listdir(dtfolder)#数据集以及数据及的实体名称
    for file_paths in data_reference:#获取从data——reference里导入的文件的子文件路径
        # print(file_paths)
        if int(file_paths.split('/')[-1].split(".")[0].split("_")[-1]) <= located:
            flist.append(os.path.join(dtfolder,file_paths))
    print(flist)
    
    traindata = torch.from_numpy(h5py.File(flist[0])[mode][:].transpose(2, 1, 0))
    for file in flist[1:]:
        if int(file.split('/')[-1].split('.')[0].split('_')[-1]) <= located:
            traindata = torch.cat((traindata, torch.from_numpy(h5py.File(file)[mode][:].transpose(2, 1, 0))), dim=0)

    print(traindata.size())
    return traindata
# get_data(r'./datasets/det/train','traindata_tem',3)
