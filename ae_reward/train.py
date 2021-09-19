from tqdm import tqdm
import torch
import torch.nn as nn
import pdb

def mse_loss(pred_r, gt_r):
    loss_fun = nn.MSELoss()
    loss = loss_fun(pred_r.squeeze(1), gt_r)
    return loss

def train_reward(dataloader, model, optimizer, gpu=False, valid=False):
    # for each batch
    loss_store = []
    if valid:
        model.eval()
    else:
        model.train()
    for batch_id, batch in enumerate(dataloader):
        # get data
        param = batch['param']
        reward = batch['reward']
        if gpu:
            param = param.cuda()
            reward = reward.cuda()
        # forward and train
        pred_r = model.forward_reward(param)
        loss = mse_loss(pred_r, reward)
        # backward
        if not valid:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # store
        loss_store.append(loss.detach().item())
    # print(pred_r[0])
    # print(reward[0])
    return loss_store

def train_model(dataloader, model, optimizer, gpu=False, valid=False):
    # for each batch
    loss_store = []
    if valid:
        model.eval()
    else:
        model.train()
    for batch_id, batch in enumerate(dataloader):
        # get data
        param = batch['param']
        reward = batch['reward']
        if gpu:
            param = param.cuda()
            reward = reward.cuda()
        # forward and train
        _, _, pred_r = model.forward(param)
        loss = mse_loss(pred_r, reward)
        # backward
        if not valid:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # store
        loss_store.append(loss.detach().item())
    return loss_store

def train_ae(dataloader, autoencoder, optimizer, gpu=False, valid=False, vae=False):
    # for each batch
    loss_store = []
    if valid:
        autoencoder.eval()
    else:
        autoencoder.train()
    for batch_id, batch in enumerate(dataloader):
        # get data
        param = batch['param']
        if gpu:
            param = param.cuda()
        # forward and train
        if vae:
            loss, rec_loss, kl_loss, _, rec_x = autoencoder.forward(param)
            loss_store.append([loss.detach().item(), rec_loss.detach().item(), kl_loss.detach().item()])
        else:
            _, rec_x = autoencoder.forward(param)
            loss = mse_loss(param, rec_x)
            # store
            loss_store.append(loss.detach().item())
        # backward
        if not valid:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return loss_store

def train_gae(dataloader, autoencoder, optimizer, gpu=False, valid=False, vae=False):
    # for each batch
    loss_store = []
    if valid:
        autoencoder.eval()
    else:
        autoencoder.train()
    for batch_id, batch in enumerate(dataloader):
        # get data
        cons_param = batch['cons_param'].float()
        disc_param = batch['disc_param'].long()
        if gpu:
            cons_param = cons_param.cuda()
            disc_param = disc_param.cuda()
        # forward and train
        mse_loss, ce_loss, _, _, _ = autoencoder.forward(cons_param, disc_param)
        loss = 10 * mse_loss + ce_loss
        loss_store.append([loss.detach().item(), mse_loss.detach().item(), ce_loss.detach().item()])
        # backward
        if not valid:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return loss_store
