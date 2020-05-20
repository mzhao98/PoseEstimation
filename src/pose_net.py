from dependencies import *


class PoseNet(nn.Module):
	def __init__(self):
		super(PoseNet, self).__init__()
		# torch.Size([256, 3, 96, 96])
		# 3 input image channel (RGB), #6 output channels, 4x4 kernel 
		self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, 
							   padding=1, dilation=1, groups=1, 
							   bias=True, padding_mode='reflect')
		
		self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=1, 
							   padding=1, dilation=1, groups=1, 
							   bias=True, padding_mode='reflect')
		
		self.conv3 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=1, 
							   padding=1, dilation=1, groups=1, 
							   bias=True, padding_mode='reflect')
		
		self.drop1 = nn.Dropout(p=0.1)
		
		self.fc1 = nn.Linear(18432, 500)
		self.fc2 = nn.Linear(500, 96)
		self.fc3 = nn.Linear(96, 32)
		
	def forward(self, x):
		# Max pooling over a (2, 2) window
		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#         print(x.shape)
		x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
#         print(x.shape)
		x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
#         print(x.shape)
		
		x = torch.flatten(x, 1)
#         print(x.shape)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.drop1(x)
		
		x = self.fc2(x)
		x = F.relu(x)
		x = self.drop1(x)
		
		x = self.fc3(x)
		x = F.relu(x)
		x = self.drop1(x)
		
		output = x
#         output = F.log_softmax(x, dim=1)
		return output
		
		