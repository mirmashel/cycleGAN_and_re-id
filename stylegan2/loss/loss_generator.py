import torch
import torch.nn as nn
import imp
import torchvision
from torchvision.models import vgg19
from network.model import Cropped_VGG19


class LossCnt(nn.Module):
    def __init__(self, device):
        super(LossCnt, self).__init__()
        
        self.VGG19 = vgg19(pretrained=True)
        self.VGG19.eval()
        self.VGG19.to(device)

        self.l1_loss = nn.L1Loss()
        self.conv_idx_list = [2,7,12,21,30] #idxes of conv layers in VGG19 cf.paper

    def forward(self, x, x_hat, vgg19_weight=1.5e-1):        

        """Retrieve vggface feature maps"""
        #define hook
        def vgg_x_hook(module, input, output):
            output.detach_() #no gradient compute
            vgg_x_features.append(output)
        def vgg_xhat_hook(module, input, output):
            vgg_xhat_features.append(output)
            
        vgg_x_features = []
        vgg_xhat_features = []

        vgg_x_handles = []
        
        conv_idx_iter = 0
        
        
        #place hooks
        for i, m in enumerate(self.VGG19.features.modules()):
            if i == self.conv_idx_list[conv_idx_iter]:
                if conv_idx_iter < len(self.conv_idx_list) - 1:
                    conv_idx_iter += 1
                vgg_x_handles.append(m.register_forward_hook(vgg_x_hook))

        #run model for x
        with torch.no_grad():
            self.VGG19(x)

        #retrieve features for x
        for h in vgg_x_handles:
            h.remove()
                        
        vgg_xhat_handles = []
        conv_idx_iter = 0
        
        #place hooks
        with torch.autograd.enable_grad():
            for i,m in enumerate(self.VGG19.features.modules()):
                if i == self.conv_idx_list[conv_idx_iter]:
                    if conv_idx_iter < len(self.conv_idx_list)-1:
                        conv_idx_iter += 1
                    vgg_xhat_handles.append(m.register_forward_hook(vgg_xhat_hook))
            self.VGG19(x_hat)
        
            #retrieve features for x
            for h in vgg_xhat_handles:
                h.remove()
        
        loss19 = 0
        for x_feat, xhat_feat in zip(vgg_x_features, vgg_xhat_features):
            loss19 += self.l1_loss(x_feat, xhat_feat)

        loss = vgg19_weight * loss19

        return loss


class LossAdv(nn.Module):
    def __init__(self, FM_weight=1e1):
        super(LossAdv, self).__init__()
        self.l1_loss = nn.L1Loss()
        
    # def forward(self, r_hat, D_res_list, D_hat_res_list):
    #     lossFM = 0
    #     for res, res_hat in zip(D_res_list, D_hat_res_list):
    #         lossFM += self.l1_loss(res, res_hat)
        
    #     return lossFM * self.FM_weight - r_hat.mean() 

    def forward(self, r_hat):
        
        return -r_hat.mean() 

    
class LossG(nn.Module):
    """
    Loss for generator meta training
    Inputs: x, x_hat, r_hat, D_res_list, D_hat_res_list, e, W, i
    output: lossG
    """
    def __init__(self, device, lambda_l1 = 1., lambda_cnt = 1.):
        super(LossG, self).__init__()
        
        self.lambda_l1 = lambda_l1
        self.lambda_cnt = lambda_cnt
        self.lambda_adv = lambda_adv

        self.lossCnt = LossCnt(device)
        self.lossAdv = LossAdv()
        self.l1_loss = nn.L1Loss()
        
    def forward(self, img_to, img_from, img_from_hat, r_hat):
        loss_cnt = self.lossCnt(img_from, img_from_hat) * self.lambda_cnt
        loss_l1 = self.l1_loss(img_from, img_from_hat) * self.lambda_l1
        # loss_adv = self.lossAdv(r_hat, D_res_list, D_hat_res_list) * self.lambda_adv
        loss_adv = self.lossAdv(r_hat) * self.lambda_adv

        return loss_cnt + loss_l1 + loss_adv
