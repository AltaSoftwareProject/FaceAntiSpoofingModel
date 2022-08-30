import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, Normalize, Resize, RandomRotation
import numpy as np
from torch.utils.data import DataLoader
from Dataset import PixWiseDataset
from Model import DeePixBiS
from Loss import PixWiseBCELoss
from Metrics import predict, test_accuracy, test_loss
from Trainer import Trainer

model = DeePixBiS()
model.load_state_dict(torch.load('./gpumodel_2.pth'))

loss_fn = PixWiseBCELoss()

opt = torch.optim.Adam(model.parameters(), lr=0.000015)

train_tfms = Compose([Resize([224, 224]),
                      RandomHorizontalFlip(),
                      RandomRotation(10),
                      ToTensor(),
                      Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

test_tfms = Compose([Resize([224, 224]),
                     ToTensor(),
                     Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

train_dataset = PixWiseDataset('./data_test.csv', transform=train_tfms)
train_ds = train_dataset.dataset()

val_dataset = PixWiseDataset('./data_training.csv', transform=test_tfms)
val_ds = val_dataset.dataset()

batch_size = 10
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)

# for x, y, z in val_dl:
# 	_, zp = model(x)
# 	print(zp)
# 	print (z)
# 	break

# print(test_accuracy(model, train_dl))
# print(test_loss(model, train_dl, loss_fn))

# 5 epochs ran
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trainer = Trainer(train_dl, val_dl, model, 10, opt, loss_fn, device)

print('Training Beginning\n')
trainer.fit()
print('\nTraining Complete')
torch.save(model.state_dict(), './gpumodel_2.pth')
