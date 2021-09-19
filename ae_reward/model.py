from ae_reward.autoencoder import AEModel, VAEModel
from ae_reward.reward_network import RNet

import torch.nn as nn

class AEReward(nn.Module):
    # init
    def __init__(self, gpu, vae=False, lat_var_dim=3):
        super(AEReward, self).__init__()

        self.vae = vae
        if not vae:
            self.ae = AEModel(gpu, lat_var_dim)
        else:
            self.ae = VAEModel(gpu, lat_var_dim=lat_var_dim)
        self.rnet = RNet(gpu)

    # forward
    def forward(self, x):
        # reconstruct
        latent_variable, rec_x = self.ae(x)
        pred_r = self.rnet(rec_x)

        return latent_variable, rec_x, pred_r

    # forward for reward network
    def forward_reward(self, x):
        pred_r = self.rnet(x)
        return pred_r

    # forward for decoder
    def forward_decode(self, lat_var):
        rec_x = self.ae.forward_decode(lat_var)
        return rec_x