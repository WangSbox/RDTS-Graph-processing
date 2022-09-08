# -*- coding: utf-8 -*-
from __future__ import print_function
import os
# from naie.datasets import get_data_reference, data_reference
import random as rand
import time
import argparse

# import itertools
# import copy, csv, math
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

import pandas as pd
import numpy as np

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from graphdataset import TempDataset
import Data_gt
from graphmodel import *
from get_edge import *
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def model_abnormal(args, criterion, traindata_tem, traindata_label, testdata_tem, testdata_label):

    torch.backends.cudnn.enabled, torch.backends.cudnn.benchmark, CUDA_LAUNCH_BLOCKING = True, True, 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCN3()
    model.to(device)

    torch.cuda.manual_seed(100), torch.manual_seed(100), np.random.seed(100), rand.seed(100)
    w, max_acc = torch.randn(100, 100), args.tgac
    nn.init.kaiming_normal_(w, mode='fan_in', nonlinearity='relu')

    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * 4 / 1000 / 1000))
    time.sleep(0.5)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=20,eta_min=1e-15)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.1, threshold_mode='rel', cooldown=0, min_lr=0, eps=2.5e-5)

    max_acc1, acc, acct, total, tal = max_acc, 0.0, 0.0, testdata_label.size(0), traindata_label.size(0)
    acc2, act2tem = 0.0, 0.9

    edge_index = get_edge_index()
    # dataset = Data(x=torch.randn(100,3),edge_index=edge_index,y=torch.rand(100,1))

    train_set = TempDataset(traindata_tem, traindata_label, edge_index)
    test_set = TempDataset(testdata_tem, testdata_label, edge_index)

    train_data = DataLoader(dataset=train_set, batch_size=args.bs, shuffle=True, num_workers=args.nw, pin_memory=False)
    test_data = DataLoader(dataset=test_set, batch_size=args.bs*2, shuffle=False, num_workers=args.nw, pin_memory=False)

    trainlosslist, testlosslist = [], []
    traintol, trainacc, trainacclist = traindata_tem.size(0), 0.0, []
    testol, testacc, testacclist = testdata_tem.size(0), 0.0, []

    with tqdm(range(args.epoch), ncols=120) as t:
        for e in t:
            test_loss, all_loss = 0.0, 0.0
            with torch.set_grad_enabled(True):
                model.train()
                loss, right, right2 = 0.0, 0, 0
                for trainframe, dataset1 in enumerate(train_data):
                    # print(trainframe)
                    optimizer.zero_grad()
                    out = model(dataset1.to(device)).squeeze()
                    loss = F.binary_cross_entropy(out, dataset1.y.squeeze())
                    loss.backward()
                    optimizer.step()
                    all_loss += loss.item()
                    if trainframe % 2000 == 0:
                        t.set_postfix(traininternum=trainframe)
                del dataset1, out
                trainlosslist.append(all_loss)
                torch.cuda.empty_cache()
            model.eval()
            with torch.no_grad():
                # if (e+1)%2 == 0:
                correct = 0
                for testframe, data1 in enumerate(test_data):
                    pred = model(data1.to(device))
                    x = pred.reshape(-1, 100)
                    y = data1.y.reshape(-1, 100)
                    # print(data,x.size(),y.size())
                    correct += np.sum((torch.topk(torch.abs_(x - y), 1).values.detach().cpu().squeeze().numpy()) < 0.25)
                    if testframe % 200 == 0:
                        t.set_postfix(testinternum=testframe)
                testacc = correct / testol
                t.set_postfix(testacc=str(testacc)[:6])
                # print('Train Accuracy:{:.4f},Test Accuracy:{:.4f},lr=:{:.5f}'.format(trainacc,testacc,optimizer.state_dict()['param_groups'][0]['lr']))
                testacclist.append(testacc)
                del data1, pred, x, y
                torch.cuda.empty_cache()
            if testacc >= max_acc1:
                max_acc1 = testacc
                torch.save(model.state_dict(), os.path.join('./detect_model/', str(testacc)[:7] + ".pth"))
                if len(os.listdir(os.path.join('./detect_model/'))) >= 20:
                    os.remove(os.listdir(os.path.join('./detect_model/'))[0])
            scheduler.step(all_loss)
            print('Train loss:{:.4f},Test Accuracy:{:.4f},lr=:{:.5f}'.format(all_loss, testacc, optimizer.state_dict()['param_groups'][0]['lr']))
    test = pd.DataFrame(columns=['test_accuracy'], data=testacclist)
    test.to_csv('/detect_model/test_accuracy.csv', index=0)
    
    # test = pd.DataFrame(columns='train_accuracy',data=trainacclist)
    # test.to_csv(os.path.join(Context.get_result_path(),'train_accuracy.csv'))

    test = pd.DataFrame(columns=['train_loss'], data=trainlosslist)
    test.to_csv('/detect_model/train_loss.csv', index=0)


    model.cpu()
    torch.cuda.empty_cache()
    time.sleep(5)
    del model
    return max_acc if max_acc1 < max_acc else max_acc1

def main(args):

    print('准备加载数据！')
    traindata_tem = Data_gt.get_data(r'./datasets/det/train', 'traindata_tem', located=args.lc)
    # traindata_tem = torch.randn(1000,100,4)
    traindata_label = torch.round_(traindata_tem[:, :, 3])
    traindata_tem = traindata_tem[:, :, :3]

    testdata_tem = Data_gt.get_data(r'./datasets/det/test', 'testdata_tem', located=args.lc)
    # testdata_tem = torch.randn(1000,100,4)
    testdata_label = torch.round_(testdata_tem[:, :, 3])
    testdata_tem = testdata_tem[:, :, :3]

    print('数据加载完成！')
    print(testdata_tem.size(), testdata_label.size())
    # torch.round(testdata_label)
    print(traindata_tem.size(), traindata_label.size())
    # torch.round(traindata_label)

    criterion = nn.BCELoss()

    print('开始训练模型！')
    max_acc = model_abnormal(args, criterion, traindata_tem, traindata_label, testdata_tem, testdata_label)
    print('模型训练完毕！获得最大训练正确率:{}'.format(max_acc))
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Extracted and Detected abnormal of data")
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--nw', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-2, help='learn_ratio')
    parser.add_argument('--lc', type=int, default=2, help='location')
    parser.add_argument('--tgac', type=float, default=0.9, help='target accuracy')
    args = parser.parse_args()
    print(args)

    if torch.cuda.is_available():
        print(torch.version.cuda)
    else:
        print('no cuda')

    main(args)
