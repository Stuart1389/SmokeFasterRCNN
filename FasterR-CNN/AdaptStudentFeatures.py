import torch.nn as nn
import torch.nn.functional as F


class AdaptFeatures(nn.Module):

	def __init__(self,input_channel=256, output_channel=1024,kernel_size=1,padding=0):
		super(AdaptFeatures, self).__init__()

		self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, padding=padding)
		self.relu = nn.ReLU()


	def forward(self, x):
		x = self.conv1(x)
		x = self.relu(x)

		return x
