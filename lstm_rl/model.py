import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pdb

class Controller(nn.Module):
    # init
    def __init__(self, embed_dim=4, hidden_dim=10, lstm_layer=2, cont_num=0, disc_table=None, gpu=False, var_init=0.1, bl_dec=0.99):
        super(Controller, self).__init__()

        # store
        self.var_init = var_init
        self.bl_dec = bl_dec
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.lstm_layer = lstm_layer
        self.cont_num = cont_num
        self.disc_table = disc_table
        if disc_table is not None:
            self.disc_num = len(disc_table)
        else:
            self.disc_num = 0
        # initialize model
        self.init_model()

    # init model
    def init_model(self):
        # discrete
        if self.disc_num > 0:
            in_dims = sum([len(self.disc_table[disc_id]) for disc_id in self.disc_table])
            self.embedder = nn.Embedding(in_dims, self.embed_dim)

        # controller
        self.control = nn.LSTM(self.embed_dim, self.hidden_dim, self.lstm_layer, batch_first=True)

        # classifier
        self.cls_list = []
        for disc_id in range(self.disc_num):
            cur_cls = nn.Sequential(
                nn.Linear(self.hidden_dim, len(self.disc_table[disc_id])),
                nn.ReLU(True),
                nn.Linear(len(self.disc_table[disc_id]), len(self.disc_table[disc_id]))
            )
            self.cls_list.append(cur_cls)
        self.cls_list = nn.Sequential(*self.cls_list)

        # regression
        self.reg = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, self.cont_num)
        )

        # logstd
        self.logstd = nn.Parameter(torch.ones(1, self.cont_num) * np.log(self.var_init), requires_grad=True)
        self.baseline = None

        #useful function
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    # forward
    def forward(self, sample_num=1):
        # init
        softmax = nn.Softmax(dim=-1)
        disc_result = []
        probs_result = []
        x = torch.zeros(1, 1, self.embed_dim)
        h0 = torch.zeros(self.lstm_layer, 1, self.hidden_dim)
        c0 = torch.zeros(self.lstm_layer, 1, self.hidden_dim)
        # forward first layer
        out, (hn, cn) = self.control(x, (h0, c0))
        logits = softmax(self.cls_list[0](out.squeeze(1)))
        print(logits)
        last_disc = torch.multinomial(logits, sample_num, replacement=True).squeeze(0).unsqueeze(1)
        hn = hn.repeat(1, sample_num, 1)
        cn = cn.repeat(1, sample_num, 1)
        disc_result.append(last_disc.clone())
        probs_result.append(torch.cat([logits[0, last_disc[i,0]].unsqueeze(0).unsqueeze(0) for i in range(sample_num)], dim=0))
        # forward for further discrete parameters
        cur_num = 0
        logits_check = logits.detach().clone()
        for disc_id in range(1, self.disc_num):
            # embed
            last_disc += cur_num
            cur_num += len(self.disc_table[disc_id-1])
            embed_vector = self.embedder(last_disc)
            # decide
            out, (hn, cn) = self.control(embed_vector, (hn, cn))
            logits = softmax(self.cls_list[disc_id](out.squeeze(1)))
            print(logits)
            last_disc = torch.multinomial(logits, 1, replacement=True)
            disc_result.append(last_disc.clone())
            probs_result.append(torch.cat([logits[i, last_disc[i,0]].unsqueeze(0).unsqueeze(0) for i in range(sample_num)], dim=0))
        # forward for regression network
        last_disc += cur_num
        embed_vector = self.embedder(last_disc)
        out, (hn, cn) = self.control(embed_vector, (hn, cn))
        reg_result = self.reg(out.squeeze(1))
        reg_result = self.sigmoid(reg_result)
        # sample for reg_result
        sample_result = self.sample_trick(reg_result)
        # concatnate discrete parameters
        disc_result = torch.cat(disc_result, dim=-1)
        probs_result = torch.cat(probs_result, dim=-1)

        return disc_result, probs_result, reg_result, sample_result, logits_check

    # calculate loss
    def cal_loss(self, disc_probs, cons_mu, sample_result, rewards):
        # check shape
        loss = torch.sum(torch.log(disc_probs), dim=1)
        zs = (sample_result.detach() - cons_mu) / torch.exp(self.logstd)
        ll_loss = -(0.5 * torch.sum(zs**2, dim=1) + torch.sum(self.logstd) + 0.5 * self.cont_num * torch.log(2*torch.tensor(np.pi)))
        # loss += (sample_result.detach() - cons_mu).pow(2) * rewards.unsqueeze(1) / var ** 2
        if self.baseline is None:
            self.baseline = torch.mean(rewards)
        self.baseline -= (1-self.bl_dec) * (self.baseline - torch.mean(rewards))
        print('current baseline is: {}'.format(self.baseline))
        loss = (loss + ll_loss) * (rewards - self.baseline)
        loss = -torch.sum(loss) / disc_probs.shape[0]
        return loss

    # sample trick
    def sample_trick(self, mu):
        # sample_data = torch.zeros_like(mu).normal_(0, var)
        # sample_data = torch.clip(mu + sample_data, 0.0, 1.0)
        noise = torch.zeros_like(mu).normal_(0, 1)
        std = torch.exp(self.logstd).expand_as(noise)
        sample_data = mu + std * noise
        return sample_data