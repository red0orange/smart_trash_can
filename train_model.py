# -*- coding: utf-8 -*-
# @Time    : 2021/4/7 上午11:06
# @Author  : red0orange
# @File    : train_model.py
# @Content : 训练pytorch模型
import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchnet import meter
from datasets import Dataset
import pretrainedmodels
from tqdm import tqdm
import numpy as np
import pandas as pd


class opt:
    checkpoint_path = None
    model_name = 'squeezenet1_0'
    data_path = 'data'
    checkpoint_save_path = 'checkout_point'
    csv_savePath = 'csv/result.csv'
    batch_size = 40
    num_workers = 4
    lr = 0.00005
    lr_decay = 0.95
    device = 'cuda'
    epoch = 100
    num_class = 3


if __name__ == '__main__':
    # if opt.checkpoint_path:
    #     model = pretrainedmodels.__dict__[opt.model_name](num_classes=1000)
    #     model.load_state_dict(torch.load(opt.checkpoint_path))
    # else:
    #     model = pretrainedmodels.__dict__[opt.model_name](num_classes=1000,pretrained='imagenet')

    if opt.checkpoint_path:
        model = getattr(torchvision.models, opt.model_name)(False)
        model.load_state_dict(torch.load(opt.checkpoint_path))
    else:
        model = getattr(torchvision.models, opt.model_name)(True)

    model.to(opt.device)
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    img_size = (224, 224)
    train_data = Dataset(opt.data_path, train=True, test=False, mean=mean, std=std, img_size=img_size)
    val_data = Dataset(opt.data_path, train=False, test=False, mean=mean, std=std, img_size=img_size)
    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # vis = Visualizer('default', port=8097)
    loss_meter = meter.AverageValueMeter()
    val_loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(opt.num_class)
    val_confusion_matrix = meter.ConfusionMeter(opt.num_class)

    last_loss = 10e5
    all_epoch = []
    all_train_loss = []
    all_val_loss = []
    all_correct = []
    best_correct = 0
    for epoch in range(opt.epoch):
        loss_meter.reset()
        val_loss_meter.reset()
        for ii, (_, data, label) in tqdm(enumerate(train_dataloader)):
            input = data.to(opt.device)
            target = label.to(opt.device)

            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.item())
            confusion_matrix.add(score[:, :opt.num_class].detach(), target.detach())

        # val the model
        model.eval()
        val_predict_true = 0
        val_predict_all = 0
        for ii, (_, data, label) in tqdm(enumerate(val_dataloader)):
            input = data.to(opt.device)
            target = label.to(opt.device)
            score = model(input)
            loss = criterion(score, target)
            score = score.cpu()
            val_confusion_matrix.add(score[:, :opt.num_class].detach(), target.detach())
            target = target.cpu().numpy()
            val_loss_meter.add(loss.item())
            predict = torch.argmax(score, dim=1).numpy()
            val_predict_all += predict.shape[0]
            val_predict_true += np.sum(target == predict)
        print('epoch: {}'.format(epoch))
        print('correct: {}'.format(val_predict_true / val_predict_all))
        print('train_loss: {}'.format(loss_meter.value()[0]))
        print('valid_loss: {}'.format(val_loss_meter.value()[0]))
        model.train()
        if loss_meter.value()[0] > last_loss:
            opt.lr = opt.lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt.lr
        if os.path.exists(os.path.join(opt.checkpoint_save_path, '{}.pth'.format(epoch - 2))):
            os.remove(os.path.join(opt.checkpoint_save_path, '{}.pth'.format(epoch - 2)))
        torch.save(model.state_dict(), os.path.join(opt.checkpoint_save_path, '{}.pth'.format(epoch)))
        if val_predict_true / val_predict_all > best_correct:
            torch.save(model.state_dict(), os.path.join(opt.checkpoint_save_path, '{}_best.pth'.format(opt.model_name)))
            best_correct = val_predict_true / val_predict_all
        last_loss = loss_meter.value()[0]
        all_epoch.append(epoch)
        all_correct.append(val_predict_true / val_predict_all)
        all_train_loss.append(loss_meter.value()[0])
        all_val_loss.append(val_loss_meter.value()[0])
        dataframe = pd.DataFrame(
            {'epoch': all_epoch, 'correct': all_correct, 'train_loss': all_train_loss, 'val_loss': all_val_loss})
        dataframe.to_csv(opt.csv_savePath, index=False, sep=',')
