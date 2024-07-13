import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os

transform = transforms.Compose(
      [
            transforms.Resize((32,32)),
            transforms.ToTensor(),
      ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)

Sample,_ = next(iter(data_loader))
Sample = Sample[0]
Input = transforms.ToPILImage()(Sample)
Input.save("InputRGB.png")
Sample = transforms.Grayscale(3)(Sample)
Sample = Sample.to(device)
Sample = Sample/255

Input = Sample*255
Input = transforms.ToPILImage()(Input)
Input.save("InputGRAY_SCALE.png")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.Encoder = self.Encoder_forward()
        self.Mean = nn.Linear(4096,4096)
        self.Std = nn.Linear(4096,4096)
        self.Decoder = self.Decoder_forward()



    def Encoder_forward(self):
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(1),
            nn.Linear(256*4*4 ,4096),
            nn.ReLU(),
        )
    def Decoder_forward(self):
         return nn.Sequential(
             nn.ReLU(),
             nn.Linear(4096,256*4*4),
             nn.ReLU(),
             nn.Unflatten(1,(256,4,4)),
             nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
             nn.ReLU(),
             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
             nn.ReLU(),
             nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
             nn.Sigmoid(),
        )

    def forward(self, X):
        X = self.Encoder(X)
        Mean = self.Mean(X)
        Var = self.Std(X)
        epsilon = torch.randn_like(Var).to(device)
        Z = Mean + epsilon * Var
        X = self.Decoder(Z)
        return X, Mean, Var


net = Net().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.0001)
CUDA_LAUNCH_BLOCKING = 1
def loss(Y_pred, Y_true, Mean , Var):
    image_loss = F.binary_cross_entropy(Y_pred, Y_true, reduction='sum')
    KLD = - 0.5 * torch.sum(1 + Var - Mean.pow(2) - Var.exp())
    return image_loss + KLD


for epoch in range(3000):
    for batch_idx, (X, _) in enumerate(data_loader):
        X = X.to(device)
        X = X/255
        Y = X.clone().to(device).detach()
        X = transforms.Grayscale(3)(X)
        Y_pred, Mean, Var = net(X)
        Loss = loss(Y_pred, Y, Mean, Var)
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Epoch:[{epoch}/{100}]  Batch:[{batch_idx}] Loss D: {Loss:.4f}")
            with torch.no_grad():
                Result,_,_ = net(torch.unsqueeze(Sample, 0))
                Result = Result*255
                Result = transforms.ToPILImage()(Result[0])
                Result.save('Result.png')






