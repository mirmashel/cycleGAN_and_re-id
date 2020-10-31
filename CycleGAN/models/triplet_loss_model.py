import torch.nn as nn
import torchvision

class TripletLossModel(nn.Module):

	def __init__(self):
		super(TripletLossModel, self).__init__()
		self.backbone = nn.Sequential(*list(torchvision.models.resnet34(pretrained = True).children())[:-1])
		self.triplet_loss = nn.TripletMarginLoss(margin = 2)


	def forward(self, anchor, positive, negative):
		self.anchor_emb = self.backbone(anchor)
		self.positive_emb = self.backbone(positive)
		self.negative_emb = self.backbone(negative)
		return self.anchor_emb, self.positive_emb, self.negative_emb

	def get_loss(self):
		return self.triplet_loss(self.anchor_emb, self.positive_emb, self.negative_emb)