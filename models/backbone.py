import torch 
import torch.nn as nn 
import torchvision


class Densenet(nn.Module):
	def __init__(self, input_size = (256, 256), N_class = 5):
		super().__init__()
		self.model = torchvision.models.densenet121(pretrained= True) #torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
		#print(self.model.features.transition2)
		self.model.features.transition2.pool = torch.nn.AvgPool2d(1,1,padding=0)
		self.model.features.transition3.pool = torch.nn.AvgPool2d(1,1,padding=0)
		#self.model.features.denseblock4
		#self.model.classifier = torch.nn.Linear(in_features = 1024, out_features = N_class, bias = True)
		#self.model.activatef = torch.nn.Sigmoid()

		self.model1 = self.model.features

	def forward(self, x):
		return self.model1(x)



	
if __name__ == '__main__':
	model = Densenet()
	#print(model)
	x = model(torch.ones(size=(3,3,224,224)))
	print(x.shape)