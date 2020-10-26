import torch.nn as nn
import os.path as osp
import os
import torch
from models.res2net import Res2Net
from models.res2net import Bottle2neck

class ReIDModel(nn.Module):
    """Basic neural network model for reidentification. Based on Res2Net"""
    
    def __init__(self, person_number):
        super(ReIDModel, self).__init__()

        path = './models/res2net50_26w_4s-06e79181.pth'
        self.backbone = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth = 26, scale = 4)
        self.backbone.load_state_dict(torch.load(path))

        self.conv_ident1 = nn.Conv2d(4096, person_number, 1)
        self.conv_ident2 = nn.Conv2d(4096, person_number, 1)
        self.conv_verif = nn.Conv2d(4096, person_number, 1)
        self.fc1 = nn.Linear(person_number, out_features=person_number)
        self.fc2 = nn.Linear(person_number, out_features=person_number)
        self.fc3 = nn.Linear(person_number, out_features=2)
        
    def forward(self, x, y):
        f1 = self.backbone(x)
        f2 = self.backbone(y)
        x = self.fc1(self.conv_ident1(f1).view(x.size(0), -1))
        y = self.fc2(self.conv_ident2(f2).view(y.size(0), -1))
        z = (f1 - f2) ** 2
        z = self.fc3(self.conv_verif(z).view(z.size(0), -1))    
        return z, x, y

    def refresh_number_of_labels(self, person_number):
        self.conv_ident1 = nn.Conv2d(4096, person_number, 1)
        self.conv_ident2 = nn.Conv2d(4096, person_number, 1)
        self.conv_verif = nn.Conv2d(4096, person_number, 1)
        self.fc1 = nn.Linear(person_number, out_features=person_number)
        self.fc2 = nn.Linear(person_number, out_features=person_number)
        self.fc3 = nn.Linear(person_number, out_features=2)

    def get_backbone_model(self):
        return self.backbone

    def save_model(self, save_path, suffix = "latest"):
        torch.save(self.backbone.state_dict(), osp.join(save_path, "backbone_{}.pth".format(suffix)))

        torch.save(self.conv_ident1.state_dict(), osp.join(save_path, "conv_ident1_{}.pth".format(suffix)))
        torch.save(self.conv_ident2.state_dict(), osp.join(save_path, "conv_ident2_{}.pth".format(suffix)))
        torch.save(self.conv_verif.state_dict(), osp.join(save_path, "conv_verif_{}.pth".format(suffix)))
        torch.save(self.fc1.state_dict(), osp.join(save_path, "fc1_{}.pth".format(suffix)))
        torch.save(self.fc2.state_dict(), osp.join(save_path, "fc2_{}.pth".format(suffix)))
        torch.save(self.fc3.state_dict(), osp.join(save_path, "fc3_{}.pth".format(suffix)))

    def load_model(self, load_path, only_backbone = False, suffix = 'latest'):
        self.backbone.load_state_dict(torch.load(osp.join(load_path, "backbone_{}.pth".format(suffix))))

        if not only_backbone:
            self.conv_ident1.load_state_dict(torch.load(osp.join(load_path, "conv_ident1_{}.pth".format(suffix))))
            self.conv_ident2.load_state_dict(torch.load(osp.join(load_path, "conv_ident2_{}.pth".format(suffix))))
            self.conv_verif.load_state_dict(torch.load(osp.join(load_path, "conv_verif_{}.pth".format(suffix))))
            self.fc1.load_state_dict(torch.load(osp.join(load_path, "fc1_{}.pth".format(suffix))))
            self.fc2.load_state_dict(torch.load(osp.join(load_path, "fc2_{}.pth".format(suffix))))
            self.fc3.load_state_dict(torch.load(osp.join(load_path, "fc3_{}.pth".format(suffix))))


            
def get_model(num_of_labels, load_path = None, initial_suffix = 'latest', only_backbone = False, device = None):
    model = ReIDModel(num_of_labels)
    if load_path is not None:
        model.load_model(load_path, only_backbone, initial_suffix)
    if device is not None:
        model = model.to(device)
    return model