import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from model import myIFCNN
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tqdm import tqdm
from loss import ir_loss, vi_loss,ssim_loss,gra_loss
import pandas as pd 
from dataset import ImgFusionSet
from test import validate
from draw import drawLineChart
def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_root = os.path.abspath(os.path.join(os.getcwd(), "..","fusion_datasets"))  # return to last dir path
    dataSets=["MSRS","RoadScene","TNO_21","TNO_25"]  # get image data set path
    dIndex = 0
    checkName = '2'
    cudnn.benchmark = True
    batchSize = 18
    epochs = 300
    weight = [1, 1, 10, 100]
    loss = torch.Tensor().to(device)
    loss_ir = torch.Tensor().to(device)
    loss_vi = torch.Tensor().to(device)
    loss_ssim = torch.Tensor().to(device)
    loss_gra = torch.Tensor().to(device)
    lastEpoch = 0
    lossList={}
    lossArray=[]

    cropSeed = torch.random.seed()
    mtransforms = transforms.Compose([
        transforms.RandomCrop(128),
        # transforms.Resize(128),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0],std=[1])
    ])
    train_dataset = ImgFusionSet(img_dir=os.path.join(data_root, "train",dataSets[dIndex]),transir=mtransforms,transvi=mtransforms)
    # for item in train_dataset:
    #     drawPicture(item)

    test_dataset = ImgFusionSet(img_dir=os.path.join(data_root, "test",dataSets[dIndex]))

    train_loader = DataLoader(train_dataset,
                                 shuffle=True,
                                 batch_size=batchSize)
    test_loader = DataLoader(test_dataset,batch_size=1)
    iterations = epochs * len(train_loader)
    print("iterarions:{}".format(iterations))
    
    model = myIFCNN().to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4,
                           betas=(0.9, 0.999), eps=1e-8)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',verbose=True)


    weightDir = 'checkPoints'+checkName
    if os.path.exists(weightDir):
        fileList = os.listdir(weightDir)
        if 'log.csv' in fileList:
            fileList.remove('log.csv')
        if len(fileList):
            fileList.sort(key=lambda x: os.path.getctime(os.path.join(weightDir,x)))
            fileName = fileList[-1]
            filePath = os.path.join(weightDir,fileName)
            chkPoint = torch.load(filePath)
            print('load weightFile:{}'.format(filePath))
            model.load_state_dict(chkPoint['model_state_dict'])
            optimizer.load_state_dict(chkPoint['optimizer_state_dict'])
            loss = chkPoint['loss']
            lastEpoch = chkPoint['epoch']

    optimizer.lr=1e-5

    csvFile = os.path.join('checkPoints'+checkName,'log.csv')
    if os.path.exists(csvFile):
        cData =pd.read_csv(csvFile)
    else:
        cData = pd.DataFrame(index=[],
                       columns=['epoch',
                                'loss',
                                'loss_ir',
                                'loss_vi',
                                'loss_ssim',
                                'loss_gra',
                                'loss_val'
                        ])


    model.train()


    for epoch in range(lastEpoch+1,lastEpoch+epochs+1):
        print('start epoch {}: ,lr:{}'.format(epoch,optimizer.state_dict()['param_groups'][0]['lr']))
        lossSum,loss_irSum,loss_viSum,loss_ssimSum,loss_graSum=0.0,0.0,0.0,0.0,0.0
        for index, (ir, vi) in tqdm(enumerate(train_loader), total=len(train_loader)):
            ir = ir.to(device)
            vi = vi.to(device)
            out = model(ir,vi)

            loss_ir = weight[0] * ir_loss(out, ir)
            loss_vi = weight[1] * vi_loss(out, vi)
            loss_ssim = weight[2] * ssim_loss(out, ir, vi)
            loss_gra = weight[3] * gra_loss(out, ir, vi)
            loss = loss_ir + loss_vi + loss_ssim + loss_gra
            loss_ir_data = loss_ir.cpu().detach().numpy()
            loss_vi_data = loss_vi.cpu().detach().numpy()
            loss_ssim_data = loss_ssim.cpu().detach().numpy()
            loss_gra_data = loss_gra.cpu().detach().numpy()
            loss_data = loss_ir_data + loss_vi_data + loss_ssim_data + loss_gra_data
            lossSum +=loss_data
            loss_irSum +=loss_ir_data
            loss_viSum +=loss_vi_data
            loss_ssimSum +=loss_ssim_data
            loss_graSum +=loss_gra_data
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        loss_mean=lossSum/len(train_loader)
        loss_ir_mean=loss_irSum/len(train_loader)
        loss_vi_mean=loss_viSum/len(train_loader)
        loss_ssim_mean=loss_ssimSum/len(train_loader)
        loss_gra_mean=loss_graSum/len(train_loader)
        # scheduler.step(loss_data)
        lossArray.append(loss_mean)
        if len(lossArray)>10:
            lossArray.pop(0)
        print('end epoch{},loss_ir:{:.4f},loss_vi:{:.4f},loss_ssim:{:.4f},loss_gra:{:.4f},loss_total:{:.4f},mean:{:.4f},std:{:.4f}.'.format(epoch,loss_ir_mean,loss_vi_mean,loss_ssim_mean,loss_gra_mean,loss_mean,np.mean(lossArray),np.std(lossArray)))
        print('last 10 loss:',end="")
        for i in lossArray:
            print(" {:.4f}".format(i),end="")
        print("")
        if epoch%5 == 0: 
            vali_loss = validate(model,test_loader,device,weight)
            lossList[epoch]=[loss_ir_mean,loss_vi_mean,loss_ssim_mean,loss_gra_mean,loss_mean,vali_loss]
            saveCheckPoint(epoch,model,optimizer,loss,checkName)
            newRow = pd.Series([
                epoch,
                loss_mean,
                loss_ir_mean,
                loss_vi_mean,
                loss_ssim_mean,
                loss_gra_mean,
                vali_loss
            ], index=['epoch', 'loss', 'loss_ir', 'loss_vi', 'loss_ssim', 'loss_gra','loss_val'])
            cData = cData._append(newRow, ignore_index=True)
            cData.to_csv(csvFile, index=False)
        
    ylist = np.asarray(list(lossList.values()))
    drawLineChart(list(lossList.keys()),['ir','vi','ssim','gra','total','validate'],False,'loss',ylist[:,0],ylist[:,1],ylist[:,2],ylist[:,3],ylist[:,4],ylist[:,5])


def saveCheckPoint(epoch,model,optimizer,loss,checkName):
    torch.save({
        'epoch':epoch,
        'model_state_dict':model.state_dict(),
        'loss':loss,
        'optimizer_state_dict':optimizer.state_dict()
        },os.path.join('checkPoints'+checkName,'model_epoch{}.tar'.format(epoch))
    )

def drawPicture(pic):
    pic1,pic2=pic
    ndPic1=pic1.squeeze().numpy()
    ndPic2=pic2.squeeze().numpy()
    plt.subplot(1,2,1)
    plt.imshow(ndPic1,plt.cm.grey)
    plt.subplot(1,2,2)
    plt.imshow(ndPic2,plt.cm.grey)
    plt.show()

if __name__ == "__main__":
    train()