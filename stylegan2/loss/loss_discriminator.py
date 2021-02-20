import torch
import torch.nn as nn

class LossDSCreal(nn.Module):
    """
    Inputs: r
    """
    def __init__(self, gan_mode):
        super(LossDSCreal, self).__init__()
        if gan_mode == 'orig':
            self.relu = nn.ReLU()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
            self.register_buffer('real_label', torch.tensor(1))
            # self.register_buffer('fake_label', torch.tensor(0))
        self.gan_mode = gan_mode
        
    def forward(self, r):
        # loss = torch.max(torch.zeros_like(r), 1 - r)
        if self.gan_mode == 'orig':
            loss = self.relu(1.0-r)
            return loss.mean()
        elif self.gan_mode == 'lsgan':
            target_tensor = self.real_label.expand_as(r)
            return self.loss(r, target_tensor)

class LossDSCfake(nn.Module):
    """
    Inputs: rhat
    """
    def __init__(self, gan_mode):
        super(LossDSCfake, self).__init__()
        if gan_mode == 'orig':
            self.relu = nn.ReLU()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
            # self.register_buffer('real_label', torch.tensor(1))
            self.register_buffer('fake_label', torch.tensor(0))
        self.gan_mode = gan_mode
        
    def forward(self, rhat):
        # loss = torch.max(torch.zeros_like(rhat),1 + rhat)
        if self.gan_mode == 'orig':
            loss = self.relu(1.0+rhat)
            return loss.mean()
        elif self.gan_mode == 'lsgan':
            target_tensor = self.fake_label.expand_as(rhat)
            return self.loss(rhat, target_tensor)