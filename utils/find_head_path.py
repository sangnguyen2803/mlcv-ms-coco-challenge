import torch
from torchvision import models

m = models.regnet_y_800mf(weights=None)  # change model here
last = None
for name, module in m.named_modules():
    if isinstance(module, torch.nn.Linear):
        last = (name, module.in_features, module.out_features)
print(last)  # ('classifier.2', 768, 1000)
