import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class Crack_Architecture_Autoencoder(BaseNet):

    def __init__(self, rep_dim=64):
        super().__init__()

        self.rep_dim = rep_dim
        self.speedup_mode = 2
        
        self.encoder = Crack_Architecture_Encoder(self.speedup_mode, self.rep_dim)
        self.decoder = Crack_Architecture_Decoder(self.speedup_mode, self.rep_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Crack_Architecture_Encoder(BaseNet):

    def __init__(self, speedup=2, rep_dim=64): # Set speedup!!!
        super().__init__()

        self.rep_dim = rep_dim # is always n x n, where n is a square (either 64 or 128)
        self.num_features_base = 512
        self.num_features_max = 512
        self.init_res = 4
        self.speedup=speedup
        
        gpu = True
        
        self.bnorm = True
        print("Speed up mode: ", speedup)
        if speedup == 3:
            if gpu == True:
                self.conv1 = nn.Conv2d(1, 32, 3, padding=1).to('cuda')
                self.bn1 = nn.BatchNorm2d(32, eps=1e-04, affine=False).to('cuda')
                self.conv2 = nn.Conv2d(32, 16, 3, padding=1).to('cuda')
                self.bn2 = nn.BatchNorm2d(16, eps=1e-04, affine=False).to('cuda')
                self.conv3 = nn.Conv2d(16, 8, 3, padding=1).to('cuda')
                self.bn3 = nn.BatchNorm2d(8, eps=1e-04, affine=False).to('cuda')
                self.conv4 = nn.Conv2d(8, 4, 3, padding=1).to('cuda')
                if rep_dim == 128:
                    self.fc1 = nn.Linear(4 * 8 * 8, rep_dim).to('cuda')
                else:
                    self.fc1 = nn.Linear(4 * 4 * 4, rep_dim).to('cuda')
            else:
                self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
                self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
                self.conv3 = nn.Conv2d(16, 8, 3, padding=1)
                self.bn3 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
                self.conv4 = nn.Conv2d(8, 4, 3, padding=1)
                if rep_dim == 128:
                    self.fc1 = nn.Linear(4 * 8 * 8, rep_dim)
                else:
                    self.fc1 = nn.Linear(4 * 4 * 4, rep_dim)
        elif speedup == 2:
            self.conv1 = nn.Conv2d(1, 64, 3, padding=1).to('cuda')
            self.bn1 = nn.BatchNorm2d(64, eps=1e-04, affine=False).to('cuda')
            self.conv2 = nn.Conv2d(64, 32, 3, padding=1).to('cuda')
            self.bn2 = nn.BatchNorm2d(32, eps=1e-04, affine=False).to('cuda')
            self.conv3 = nn.Conv2d(32, 16, 3, padding=1).to('cuda')
            self.bn3 = nn.BatchNorm2d(16, eps=1e-04, affine=False).to('cuda')
            self.conv4 = nn.Conv2d(16, 8, 3, padding=1).to('cuda')
            if rep_dim == 128:
                self.fc1 = nn.Linear(8 * 8 * 8, rep_dim).to('cuda')
            else:
                self.fc1 = nn.Linear(8 * 4 * 4, rep_dim).to('cuda')
        else:
            self.conv1 = nn.Conv2d(1, 128, 3, padding=1).to('cuda')
            self.bn1 = nn.BatchNorm2d(128, eps=1e-04, affine=False).to('cuda')
            self.conv2 = nn.Conv2d(128, 64, 3, padding=1).to('cuda')
            self.bn2 = nn.BatchNorm2d(64, eps=1e-04, affine=False).to('cuda')
            self.conv3 = nn.Conv2d(64, 32, 3, padding=1).to('cuda')
            self.bn3 = nn.BatchNorm2d(32, eps=1e-04, affine=False).to('cuda')
            self.conv4 = nn.Conv2d(32, 16, 3, padding=1).to('cuda')
            if rep_dim == 128:
                self.fc1 = nn.Linear(16 * 8 * 8, rep_dim).to('cuda')
            else:
                self.fc1 = nn.Linear(16 * 4 * 4, rep_dim).to('cuda')
                
    def num_features(self, res):
        return min(int(self.num_features_base/self.speedup / (2 ** res)), self.num_features_max)

    def forward(self, x):
        x = self.conv1(x)
        x = F.interpolate(F.leaky_relu(self.bn1(x)), scale_factor=1/2)
        x = self.conv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2(x)), scale_factor=1/2)
        x = self.conv3(x)
        x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=1/2)
        x = self.conv4(x)
        x = F.interpolate(F.leaky_relu(x), scale_factor=1/2)
        x = x.view(x.size(0), -1)
        x = self.fc1(x) 

        return x

class Crack_Architecture_Decoder(BaseNet):

    def __init__(self, speedup=2, rep_dim=64):
        super().__init__()
        self.rep_dim = rep_dim
        self.num_features_base = 4096
        self.num_features_max = 512
        self.init_res = 4

        self.rep_dim = rep_dim

        #Decoder network
        if speedup == 3:
            if self.rep_dim == 128:
                self.deconv1 = nn.ConvTranspose2d(8, 8, 3, bias=False, padding=1).to('cuda')
            else:
                self.deconv1 = nn.ConvTranspose2d(4, 8, 3, bias=False, padding=1).to('cuda')
            self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False).to('cuda')
            self.deconv2 = nn.ConvTranspose2d(8, 16, 3, bias=False, padding=1).to('cuda')
            self.bn2 = nn.BatchNorm2d(16, eps=1e-04, affine=False).to('cuda')
            self.deconv3 = nn.ConvTranspose2d(16, 32, 3, bias=False, padding=1).to('cuda')
            self.bn3 = nn.BatchNorm2d(32, eps=1e-04, affine=False).to('cuda')
            if rep_dim == 128:
                self.deconv4 = nn.ConvTranspose2d(32, 64, 3, bias=False, padding=1).to('cuda')
            else: 
                self.deconv4 = nn.ConvTranspose2d(32, 1, 3, bias=False, padding=1).to('cuda')
            self.bn4 = nn.BatchNorm2d(64, eps=1e-04, affine=False).to('cuda')
            self.deconv5 = nn.ConvTranspose2d(64, 1, 3, bias=False, padding=1).to('cuda')
        elif speedup == 2:
            if self.rep_dim == 128:
                self.deconv1 = nn.ConvTranspose2d(8, 16, 3, bias=False, padding=1).to('cuda')
            else:
                self.deconv1 = nn.ConvTranspose2d(4, 16, 3, bias=False, padding=1).to('cuda')
            self.bn1 = nn.BatchNorm2d(16, eps=1e-04, affine=False).to('cuda')
            self.deconv2 = nn.ConvTranspose2d(16, 32, 3, bias=False, padding=1).to('cuda')
            self.bn2 = nn.BatchNorm2d(32, eps=1e-04, affine=False).to('cuda')
            self.deconv3 = nn.ConvTranspose2d(32, 64, 3, bias=False, padding=1).to('cuda')
            self.bn3 = nn.BatchNorm2d(64, eps=1e-04, affine=False).to('cuda')
            if rep_dim == 128:
                self.deconv4 = nn.ConvTranspose2d(64, 128, 3, bias=False, padding=1).to('cuda')
            else:
                self.deconv4 = nn.ConvTranspose2d(64, 1, 3, bias=False, padding=1).to('cuda')
            self.bn4 = nn.BatchNorm2d(128, eps=1e-04, affine=False).to('cuda')
            self.deconv5 = nn.ConvTranspose2d(128, 1, 3, bias=False, padding=1).to('cuda')
        else:
            if self.rep_dim == 128:
                self.deconv1 = nn.ConvTranspose2d(8, 32, 3, bias=False, padding=1).to('cuda')
            else:
                self.deconv1 = nn.ConvTranspose2d(4, 32, 3, bias=False, padding=1).to('cuda')
            self.bn1 = nn.BatchNorm2d(32, eps=1e-04, affine=False).to('cuda')
            self.deconv2 = nn.ConvTranspose2d(32, 64, 3, bias=False, padding=1).to('cuda')
            self.bn2 = nn.BatchNorm2d(64, eps=1e-04, affine=False).to('cuda')
            self.deconv3 = nn.ConvTranspose2d(64, 128, 3, bias=False, padding=1).to('cuda')
            self.bn3 = nn.BatchNorm2d(128, eps=1e-04, affine=False).to('cuda')
            if rep_dim == 128:
                self.deconv4 = nn.ConvTranspose2d(128, 256, 3, bias=False, padding=1).to('cuda')
            else:
                self.deconv4 = nn.ConvTranspose2d(128, 1, 3, bias=False, padding=1).to('cuda')
            self.bn4 = nn.BatchNorm2d(256, eps=1e-04, affine=False).to('cuda')
            self.deconv5 = nn.ConvTranspose2d(256, 1, 3, bias=False, padding=1).to('cuda')
            
    def forward(self, x):
        if self.rep_dim == 128:
            x = x.view(x.size(0), int(self.rep_dim / 16), 4, 4)
        else:
            x = x.view(x.size(0), int(self.rep_dim / 16), 4, 4)
        x = F.interpolate(F.leaky_relu(x), scale_factor=2)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn1(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2)
        x = self.deconv4(x)
        if self.rep_dim == 128:
            x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)
            x = self.deconv5(x)
        x = torch.sigmoid(x)
        return x