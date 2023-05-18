import time as ti
import torch
import numpy as np
import random
import os
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import argparse
from test import evaluation
import datasets.mvtec as mvtec
import model.cnnAndCaitModel as MultiScaleModel
from tqdm import tqdm

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def cosine(a, b):
    N, C, _ = a.shape
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += torch.sum((1-cos_loss(a[item].view(a[item].shape[0],-1), b[item].view(b[item].shape[0],-1))))
    loss = loss / N
    return loss

def parse_args():
    parser = argparse.ArgumentParser('MLDFR')
    parser.add_argument('--data_path', type=str, default= './MVTec')
    parser.add_argument('--save_path', type=str, default='./Test')
    parser.add_argument('--epoch', type=int, default = 500)
    parser.add_argument('--block_k', type=int, default = 8)
    parser.add_argument('--input_size', type=int, default = 384)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--number', type=int, default = 1)
    return parser.parse_args()

def train(_class_):
    args = parse_args()
    number = args.number
    print(_class_)
    class_name = _class_
    image_size = args.input_size
    args.save_path = args.save_path + str(number)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #loading data
    test_data = mvtec.MVTecDataset(args.data_path, class_name=class_name, is_train=False,resize=image_size, cropsize=image_size)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    train_dataset = mvtec.MVTecDataset(args.data_path, class_name=class_name, is_train=True,resize=image_size, cropsize=image_size)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    d = 736
    model = MultiScaleModel.MultiScaleModel(backbone_name='resnet18',block_k = args.block_k, chanel=d)
    model.to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.featureRec.parameters()), lr=0.001,weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)

    #save meassage
    os.makedirs(os.path.join(args.save_path), exist_ok=True)
    if args.epoch == 0:
        model.featureRec.load_state_dict(torch.load(args.save_path + '/' + '_class_' + '_' + args.arch + '_' + str(number) + '.pt'))  # ,map_location='cpu'))
        T1 = ti.time()
        auroc_sp, auroc_px, aupro_px = evaluation(model, test_dataloader, device, class_name,image_size)
        T2 = ti.time()
        print('interence time:%s second' % ((T2 - T1)))
        print(" I = {:.3f} P = {:.3f} PRO = {:.3f}".format(auroc_sp, auroc_px, aupro_px))

    Best_auroc_px, Best_auroc_sp, Best_aupro_px,Best_epoch = 0, 0, 0, 0

    for epoch in range(args.epoch):
        loss_list = []
        for (x, _, _, z) in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
            N,_,_,_ = x.shape
            inputs, outputs = model(x.to(device), z.to(device), True)
            scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, inputs.dim())))
            loss = torch.mean(scores) + 140 * cosine(inputs, outputs)
            optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            scheduler.step()

        if  (epoch + 1) % 1 == 0:
            auroc_sp, auroc_px,  aupro_px = evaluation(model, test_dataloader, device,class_name,image_size)
            # save model
            if (( auroc_px + aupro_px + auroc_sp) > (Best_auroc_px  + Best_aupro_px + Best_auroc_sp)):
                Best_auroc_px, Best_auroc_sp, Best_aupro_px = auroc_px, auroc_sp, aupro_px
                Best_epoch = epoch + 1
                torch.save(model.featureRec.state_dict(),args.save_path + '/' + class_name + '_' + str(number) + '.pt')
        print(
            "{}--:LOSS = {:.2f} I = {:.3f} P = {:.3f} PRO = {:.3f} Best-I = {:.3f} Best-P = {:.3f} Best-PRO = {:.3f} Best-epoch = {}".format(
                epoch + 1, np.mean(loss_list), auroc_sp, auroc_px,  aupro_px, Best_auroc_sp,Best_auroc_px, Best_aupro_px, Best_epoch))
        if (epoch + 1 - Best_epoch) >= 50:
            break

    return auroc_px, auroc_sp, aupro_px


if __name__ == '__main__':
    seed_torch(1024)
    item_list = ['metal_nut','toothbrush','bottle','zipper','pill','screw','hazelnut','transistor','cable','capsule','grid','wood','carpet','tile','leather',]
    #'bracket_black','tubes','connector','bracket_brown','bracket_white','metal_plate',
    for i in item_list:
        train(i)



