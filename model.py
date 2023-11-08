'''---------------------------------------------------------------------------
IFCNN: A General Image Fusion Framework Based on Convolutional Neural Network
----------------------------------------------------------------------------'''
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import cv2 as cv
import matplotlib.pyplot as plt

# My Convolution Block
class ConvBlock(nn.Module):
    def __init__(self, inplane, outplane):
        super(ConvBlock, self).__init__()
        self.padding = (1, 1, 1, 1)
        self.conv = nn.Conv2d(inplane, outplane, kernel_size=3, padding=0, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(outplane)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = F.pad(x, self.padding, 'replicate')
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out

class ResBlock(nn.Module):
    def __init__(self,in_channel,out_channel,stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride,padding=1,padding_mode='replicate', bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1,padding_mode='replicate', bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)


    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class Conv2(nn.Module):
    def __init__(self,channel,stride=1):
        super(Conv2,self).__init__()
        self.convLayers=nn.ModuleList()
        for i in range(8):
            self.convLayers.append(ResBlock(channel,channel,stride))


    def forward(self,x):
        for layer in self.convLayers:
            x = layer(x)
        return x


class IFCNN(nn.Module):
    def __init__(self, resnet):
        super(IFCNN, self).__init__()
        # self.conv1 = nn.Conv2d(1,64,kernel_size=7,padding=0,stride=1,bias=False)
        self.conv2 = Conv2(64)
        # self.conv2 = ConvBlock(64,64)
        self.conv3 = ConvBlock(64,64)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=1, padding=0, stride=1)

        # Initialize conv1 with the pretrained resnet101 and freeze its parameters

        self.fuseConv = nn.Conv2d(128,64,kernel_size=1,padding=0,bias=True,groups=64)
          # Initialize parameters for other parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d) :
                if m.groups == 64:
                    nn.init.constant_(m.weight,0.5)
                    nn.init.constant_(m.bias,0)
                else:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))

        # for p in resnet.parameters():
        #     p.requires_grad = False

        self.conv1 = resnet.conv1
        self.conv1.stride = 1
        self.conv1.padding = (0, 0)
        # print(self.conv1)
    def tensor_acum(self, tensors):
        sum_tensor = tensors[0].clone()
        sub_tensor = tensors[0].clone()
        for i in range(1,len(tensors)):
            sum_tensor = torch.add(sum_tensor,tensors[i])
            sub_tensor = torch.sub(sub_tensor,tensors[i])

        abs_tensor = torch.abs(sub_tensor)
        return sum_tensor,abs_tensor
    def operate(self, operator, tensors):
        out_tensors = []
        for tensor in tensors:
            out_tensor = operator(tensor)
            out_tensors.append(out_tensor)
        return out_tensors

    def tensor_repeat(self, tensors):
        out_tensors = []
        for tensor in tensors:
            out_tensor = tensor.repeat(1,3,1,1)
            out_tensors.append(out_tensor)
        return out_tensors

    def tensor_padding(self, tensors, padding=(1, 1, 1, 1), mode='constant', value=0):
        out_tensors = []
        for tensor in tensors:
            out_tensor = F.pad(tensor, padding, mode=mode, value=value)
            out_tensors.append(out_tensor)
        return out_tensors

    def forward(self, *tensors):
        # Feature extraction
        # outs = tensors
        outs = self.tensor_repeat(tensors)
        outs = self.tensor_padding(tensors=outs, padding=(3, 3, 3, 3), mode='replicate')
        outs = self.operate(self.conv1, outs)
        outs = self.operate(self.conv2, outs)
        drawPicture(outs[0].cpu())
        drawPicture(outs[1].cpu())
        
        # Feature fusion
        sum,abs =self.tensor_acum(outs)
        out = torch.stack((sum,abs),dim=2)
        out = torch.flatten(out,start_dim=1,end_dim=2)
        out = self.fuseConv(out)
        # drawPicture(out.cpu())

        # Feature reconstruction
        out = self.conv3(out)
        out = self.conv4(out)
        out = torch.mean(out,dim=1,keepdim=True)
        return out
    

def drawPicture(picList):
    print(picList.shape)
    picList = picList.squeeze()
    print(picList.shape)
    for i in range(len(picList)):
        newPic = picList[i].numpy()
        ax = plt.subplot(8,8,1+i)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(newPic)
    plt.subplots_adjust(wspace=0,hspace=0)
    plt.show()


def myIFCNN():
    # pretrained resnet101
    resnet = models.resnet101(pretrained=True)
    # our model
    model = IFCNN(resnet)
    return model

if __name__ == '__main__':
    test1 = torch.ones((1, 1, 2, 2), dtype=torch.float32)
    test2 = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    model = myIFCNN().cuda()
    outImg = model(test1.cuda(), test2.cuda())
    print("outputSize", outImg.shape)