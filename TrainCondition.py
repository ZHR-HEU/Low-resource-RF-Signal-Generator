

import os
from typing import Dict
import numpy as np

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy import io
from dataprocess import DataProcess as dataprocess
from DiffusionCondition import GaussianDiffusionSampler, GaussianDiffusionTrainer
from ModelCondition import UNet
from Scheduler import GradualWarmupScheduler


def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    print('==> Preparing data..')
    dataFile = 'data_5_600.mat'
    dataset_origin = dataprocess(dataFile)
    print("dataset_origin size:")
    print(dataset_origin.data.shape)
    train_data = dataset_origin.data[0:2160]
    train_data = torch.from_numpy(train_data.astype(np.float32))
    print("train_data size:")
    print(train_data.shape)
    train_label = dataset_origin.label[0:2160]
    train_label = torch.from_numpy(train_label).long()
    train_label = train_label.squeeze()
    dataset = torch.utils.data.TensorDataset(train_data, train_label)
    print("dataset len:")
    print(dataset.__len__())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=0)

    # model setup
    net_model = UNet(T=modelConfig["T"], num_labels=6, ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_dir"], modelConfig["training_load_weight"]), map_location=device), strict=False)
        print("Model weight load down.")
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=modelConfig["multiplier"],
                                             warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    # start training
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                # train
                b = images.shape[0]
                optimizer.zero_grad()
                x_0 = images.to(device)
                labels = labels.to(device) + 1
                if np.random.rand() < 0.1:
                    labels = torch.zeros_like(labels).to(device)
                loss = trainer(x_0, labels).sum() / b ** 2.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        if e % 50 == 0:
            torch.save(net_model.state_dict(), os.path.join(
                modelConfig["save_dir"], 'ckpt_' + str(e) + "_.pt"))


def eval(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    # 加载模型并评估
    with torch.no_grad():
        samples_per_label = 1  # 每个标签的样本数量
        labelList = []
        for k in range(6):
            labelList.extend([torch.ones(size=[1]).long() * k] * samples_per_label)
        labels = torch.cat(labelList, dim=0).long().to(device) + 1
        print("labels: ", labels)
        model = UNet(T=modelConfig["T"], num_labels=6, ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
        ckpt = torch.load(os.path.join(
            modelConfig["save_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], w=modelConfig["w"]).to(device)
        # 从标准正态分布中采样
        noisySignal = torch.randn(
            size=[len(labels), 2, modelConfig["signal_length"]], device=device)
        sampledSignal = sampler(noisySignal, labels)
        sampledSignal = sampledSignal * 0.5 + 0.5  # [0 ~ 1]
        # 将tensor转换为numpy ndarray并保存为.mat文件
        sampledSignal_np = sampledSignal.cpu().numpy()
        io.savemat('sampledSignal.mat', {'sampledSignal': sampledSignal_np})

