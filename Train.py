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
model.eval()
#model.load_state_dict(torch.load('./MyModel.pth'))

loss_fn = PixWiseBCELoss()

opt = torch.optim.Adam(model.parameters(), lr=0.0001)

train_tfms = Compose([Resize([224, 224]),
                      RandomHorizontalFlip(),
                      RandomRotation(10),
                      ToTensor(),
                      Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

test_tfms = Compose([Resize([224, 224]),
                     ToTensor(),
                     Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

train_dataset = PixWiseDataset('./data_training.csv', transform=train_tfms)
train_ds = train_dataset.dataset()

val_dataset = PixWiseDataset('./data_testing.csv', transform=test_tfms)
val_ds = val_dataset.dataset()

batch_size = 10
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)

trainer = Trainer(train_dl, val_dl, model, 1, opt, loss_fn)

print('Training Beginning\n')
trainer.fit()
print('\nTraining Complete')
torch.save(model.state_dict(), './MyModel.pth')