from torchvision.models import resnet50
from thop import profile
import torch
model = resnet50()
print(model)
input = torch.randn(1, 3, 224, 224)
flops, params = profile(model, inputs=(input, ))


