# -*- coding: utf-8 -*-
from __future__ import print_function
import os
# import time
import h5py
import numpy as np


def get_data(args, dtfolder, dname):
    fea_num = [0, 1, 2, 3, 4]
    if not args.ab:
        fea_num.remove(3)
    if not args.dist:
        fea_num.remove(4)
    traindata_tem = np.empty([0, 100, len(fea_num)])
    traindata_label = np.empty([0, 100])

    if dname == 'traindata_temper':   
        flist = []
        for ite in range(args.trn):
            data_reference = os.listdir(os.path.join(dtfolder + str(ite+1))) #
            if len(data_reference) <= 0:
                continue
            for file_paths in data_reference:  #获取从data——reference里导入的文件的子文件路径
                print(file_paths)
                if int(file_paths.split(".")[0].split("_")[-1]) < args.lc:  #不满足条件的直接不移动过来 小于想要训练位置的数据均不移动
                    print(os.path.join(dtfolder + str(ite+1), file_paths))
                    flist.append(os.path.join(dtfolder + str(ite+1), file_paths))
        print((flist))            
        for _, file in enumerate(flist):
            fdata = h5py.File(os.path.join(file))[dname]
            traindata_tem = np.concatenate((traindata_tem, fdata[fea_num, :, :].transpose(2, 1, 0)), axis=0)
            traindata_label = np.concatenate((traindata_label, fdata[5, :, :].transpose(1,0)), axis=0)
            del fdata
    elif dname == 'testdata_temper':
        flist = []
        data_reference = os.listdir(dtfolder)  
        for file_paths in data_reference:  #获取从data——reference里导入的文件的子文件路径
            # print(file_paths)
            if len(flist) < args.tesn:
                flist.append(os.path.join(dtfolder, file_paths))
        print(len(flist))        
        for _, file in enumerate(flist):

            fdata = h5py.File(os.path.join(file))[dname]
            for j in range(args.lc):
                traindata_tem = np.concatenate((traindata_tem,
                                            fdata[fea_num, j * 100:(j + 1) * 100, :].transpose(2, 1, 0)), axis=0)
                traindata_label = np.concatenate((traindata_label,
                                            fdata[5, j * 100:(j + 1) * 100, :].transpose(1, 0)), axis=0)
            del fdata
    else:
        raise('check your configurations')
    if args.ab:
        if args.round:
            traindata_tem[:, :, 3] = np.round_(traindata_tem[:, :, 3])  #将状态舍入
            print('将异常状态舍入')
        else:
            print('未将异常状态舍入')
    print(traindata_tem.shape, traindata_label.shape)
    return traindata_tem, traindata_label
# get_data('',r'./datasets/den/train','testdata_temper')
