from ae_reward.dense_module import DenseBlock

import torch
import torch.nn as nn

import pdb


# encoder for continuous as well as discrete parameters
class GeneralAEModel(nn.Module):
    # init
    def __init__(self, gpu=False, cont_num=0, disc_num=0, disc_table=None, embed_dim=4, latent_dim=6):
        super(GeneralAEModel, self).__init__()

        assert cont_num + disc_num > 0
        if disc_num > 0:
            assert disc_table is not None
        self.gpu = gpu
        self.cont_num = cont_num
        self.disc_num = disc_num
        self.disc_table = disc_table
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        # define network
        self.init_model()

    # init structure
    def init_model(self):
        # discrete
        if self.disc_num > 0:
            in_dims = sum([len(self.disc_table[disc_id]) for disc_id in self.disc_table])
            self.embedder = nn.Embedding(in_dims, self.embed_dim)
        # encoder
        param_dim = self.embed_dim * self.disc_num + self.cont_num
        self.encoder = DenseBlock(num_layers=5, growth_rate=32, input_size=param_dim, output_size=self.latent_dim)
        # decoder
        self.decoder = DenseBlock(num_layers=5, growth_rate=32, input_size=self.latent_dim, output_size=param_dim)
        # classifier for discrete 
        # TODO: whether together or separate
        self.cls_list = []
        for disc_id in self.disc_table:
            cur_cls = nn.Sequential(
                nn.Linear(self.embed_dim*self.disc_num, len(self.disc_table[disc_id])),
                nn.ReLU(True),
                nn.Linear(len(self.disc_table[disc_id]), len(self.disc_table[disc_id]))
            )
            self.cls_list.append(cur_cls)
        self.cls_list = nn.Sequential(*self.cls_list)
        # regression for continuous
        self.reg = nn.Sequential(
            nn.Linear(self.cont_num, self.cont_num)
        )
        # useful function
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    # embed discrete parameters
    def embed_disc(self, disc_param):
        assert disc_param.shape[1] == self.disc_num
        disc_param_in = disc_param.clone()
        # fit for embedder
        extra = 0
        for disc_id in range(1, self.disc_num):
            extra += len(self.disc_table[disc_id-1])
            disc_param_in[:, disc_id] += extra
        # embed
        disc_embed = self.embedder(disc_param_in)
        return disc_embed

    # process
    def forward(self, cons_param, disc_param):
        # init
        batch_size = cons_param.shape[0]
        # embed discrete parameters
        if self.disc_num > 0:
            disc_embed = self.embed_disc(disc_param).view(batch_size, -1)
        # concatenate with continuous parameters
        total_embed = disc_embed
        if self.cont_num > 0:
            total_embed = torch.cat([total_embed, cons_param], dim=1)
        # encode
        latent_variable = self.encoder(total_embed)
        latent_variable = self.tanh(latent_variable)
        # decode
        rec_total_param = self.decoder(latent_variable)
        rec_disc_param = rec_total_param[:, :self.embed_dim*self.disc_num]
        rec_cons_param = rec_total_param[:, self.embed_dim*self.disc_num:]
        # reconstruct
        disc_logits = [self.cls_list[disc_id](rec_disc_param) \
                        for disc_id in range(self.disc_num)]
        rec_cons_param = self.reg(rec_cons_param)
        rec_cons_param = self.sigmoid(rec_cons_param)
        # calculate loss
        mse_loss, ce_loss = self.rec_loss(rec_cons_param, cons_param, disc_logits, disc_param)
        return mse_loss, ce_loss, latent_variable, rec_cons_param, disc_logits

    # decode
    def forward_decode(self, lat_vector):
        rec_total_param = self.decoder(lat_vector)
        rec_disc_param = rec_total_param[:, :self.embed_dim*self.disc_num]
        rec_cons_param = rec_total_param[:, self.embed_dim*self.disc_num:]
        # reconstruct
        rec_cons_param = self.reg(rec_cons_param)
        rec_cons_param = self.sigmoid(rec_cons_param)
        disc_logits = [self.cls_list[disc_id](rec_disc_param) \
                        for disc_id in range(self.disc_num)]
        rec_disc_param = torch.cat([torch.max(logit, dim=-1, keepdims=True)[1] \
                        for logit in disc_logits], dim=-1)
        return rec_cons_param, rec_disc_param

    # calculate reconstruction loss
    def rec_loss(self, rec_cons_param, cons_param, disc_logits, disc_param):
        # reconstruction loss for continuous parameters
        mse_fun = nn.MSELoss()
        mse_loss = mse_fun(rec_cons_param, cons_param)
        # reconstruction loss for discrete parameters
        ce_fun = nn.CrossEntropyLoss()
        ce_loss = [ce_fun(disc_logits[disc_id], disc_param[:, disc_id]) for disc_id in range(self.disc_num)]
        ce_loss = sum(ce_loss) / len(ce_loss)
        return mse_loss, ce_loss


class AEModel(nn.Module):
    # init
    def __init__(self, gpu=False, lat_var_dim=2):
        super(AEModel, self).__init__()

        self.gpu = gpu
        self.encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(True),
            nn.Linear(32, lat_var_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(lat_var_dim, 32),
            nn.ReLU(True),
            nn.Linear(32, 4)
        )
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    
    # process
    def forward(self, x):
        latent_variable = self.encoder(x)
        latent_variable = self.tanh(latent_variable)
        rec_x = self.decoder(latent_variable)
        rec_x = self.sigmoid(rec_x)
        return latent_variable, rec_x

    def forward_decode(self, lat_var):
        rec_x = self.decoder(lat_var)
        rec_x = self.sigmoid(rec_x)
        return rec_x

class VAEModel(nn.Module):
    # init
    def __init__(self, gpu=False, beta=0.01, beta_step=0.001, beta_freq=100, warm_step = 5000, lat_var_dim=2):
        super(VAEModel, self).__init__()

        self.gpu = gpu
        self.latent_dim = lat_var_dim
        self.beta = 0.0
        self.final_beta = beta
        self.beta_step = beta_step
        self.beta_freq = beta_freq
        self.warmup_step = warm_step
        self.cur_step = 0
        self.encoder = nn.Sequential(
            nn.Linear(4, 32),
            # nn.LeakyReLU(inplace=True),
            nn.ReLU(True),
            nn.Linear(32, 32),
            # nn.LeakyReLU(inplace=True),
            nn.ReLU(True),
            nn.Linear(32, 2*self.latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 32),
            # nn.LeakyReLU(inplace=True),
            nn.ReLU(True),
            nn.Linear(32, 32),
            # nn.LeakyReLU(inplace=True),
            nn.ReLU(True),
            nn.Linear(32, 4)
        )
        self.sigmoid = nn.Sigmoid()
    
    # kl divergence
    def cal_kl_loss(self, mu, logstd):
        kl_div = 0.5 * (torch.exp(2*logstd) + mu.pow(2) - 1.0 - 2*logstd)
        kl_loss = kl_div.sum() / mu.shape[0]
        return kl_loss

    # reconstruction loss
    def cal_rec_loss(self, x, rec_x):
        loss_fun = nn.MSELoss()
        rec_loss = loss_fun(rec_x, x)
        return rec_loss

    # process
    def forward(self, x):
        # encode
        enc_out = self.encoder(x)
        mu, logstd = enc_out[:, :self.latent_dim], enc_out[:, self.latent_dim:]
        # resample trick
        lat_dist = torch.distributions.Normal(loc=mu, scale=torch.exp(logstd))
        z_sample = lat_dist.rsample()
        # decode
        rec_x = self.decoder(z_sample)
        rec_x = self.sigmoid(rec_x)
        # loss
        rec_loss = self.cal_rec_loss(x, rec_x)
        kl_loss = self.cal_kl_loss(mu, logstd)
        loss = rec_loss + self.beta * kl_loss
        # update beta
        self.cur_step += 1
        self.beta_update()

        return loss, rec_loss, kl_loss, z_sample, rec_x

    def forward_decode(self, lat_var):
        rec_x = self.decode_deterministic(lat_var)
        return rec_x

    # do decode
    def decode_deterministic(self, z:torch.Tensor) -> torch.Tensor:
        rec_x = self.decoder(z)
        rec_x = self.sigmoid(rec_x)
        return rec_x

    # beta update
    def beta_update(self):
        if self.cur_step > self.warmup_step:
            if self.cur_step % self.beta_freq == 0:
                self.beta = min(self.beta+self.beta_step, self.final_beta)
