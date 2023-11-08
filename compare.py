import os
import time
import torch
import IFCNNmodel 
import model
os.environ['CUDA_VISIBLE_DEVICES']='3'

from torchvision import transforms
from torch.autograd import Variable

from PIL import Image
import numpy as np
import pandas as pd
from dataset import ImgFusionSet
from evaluator import Evaluator
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))
# load pretrained model

def drawLineChart(dlist1,dlist2):
    xlist = ['EN','SD','SF','AG','VIFF','Qabf']
    plt.plot(xlist,dlist1,c="red",marker='o')
    plt.plot(xlist,dlist2,c="blue",marker='o')
    plt.title('metrics')
    # plt.savefig(fname='result.png')
    plt.show()


ifCNNmodel = IFCNNmodel.myIFCNN().to(device)
weightFile1 = os.path.join('checkPointsIFCNN','model_epoch60.tar')
ifCNNmodel.load_state_dict(torch.load(weightFile1)['model_state_dict'])
ifCNNmodel.eval()
mymodel =  model.myIFCNN().to(device=device)
weightFile2 = os.path.join('checkPoints','model_epoch60.tar')
mymodel.load_state_dict(torch.load(weightFile2)['model_state_dict'])
mymodel.eval()

datasets = [ 'MSRS'] # Color MultiFocus, Infrared-Visual, MeDical image datasets
# datasets = [ 'MSRS','TNO_21','RoadScene', 'MRI-PET','MRI_SPECT'] # Color MultiFocus, Infrared-Visual, MeDical image datasets
is_save = False                # if you do not want to save images, then change its value to False
testRoot = '../fusion_datasets/test/'

for j in range(len(datasets)):
    begin_time = time.time()
    setRoot = os.path.join(testRoot, datasets[j])
    test_dataset = ImgFusionSet(img_dir=setRoot)
    for k in range(1):
    # for k in range(len(test_dataset)):
        img1, img2 = test_dataset[k]
        img1.unsqueeze_(0)
        img2.unsqueeze_(0)
        print(test_dataset.getName(k))
        # perform image fusion
        en_sum1,sd_sum1,sf_sum1,ag_sum1,vif_sum1,qabf_sum1=0.0,0.0,0.0,0.0,0.0,0.0
        en_sum2,sd_sum2,sf_sum2,ag_sum2,vif_sum2,qabf_sum2=0.0,0.0,0.0,0.0,0.0,0.0
        with torch.no_grad():
            res1 = ifCNNmodel(Variable(img1.cuda()), Variable(img2.cuda()))
            res2 = mymodel(Variable(img1.cuda()), Variable(img2.cuda()))
            res1_img = res1.cpu().squeeze()
            res2_img = res2.cpu().squeeze()

            ndImg1 = img1.squeeze().numpy()
            ndImg2 = img2.squeeze().numpy()
            ndResImg1 = res1_img.numpy()
            ndResImg2 = res2_img.numpy()
            ev = Evaluator()
            ev.input_check(ndResImg1)
            en_sum1+=  ev.EN(ndResImg1)
            sd_sum1+=  ev.SD(ndResImg1)
            sf_sum1+=  ev.SF(ndResImg1)
            ag_sum1+=  ev.AG(ndResImg1)
            vif_sum1+=  ev.VIFF(ndResImg1,ndImg1,ndImg2)
            qabf_sum1+=  ev.Qabf(ndResImg1,ndImg1,ndImg2)

            ev.input_check(ndResImg2)
            en_sum2+=  ev.EN(ndResImg2)
            sd_sum2+=  ev.SD(ndResImg2)
            sf_sum2+=  ev.SF(ndResImg2)
            ag_sum2+=  ev.AG(ndResImg2)
            vif_sum2+=  ev.VIFF(ndResImg2,ndImg1,ndImg2)
            qabf_sum2+=  ev.Qabf(ndResImg2,ndImg1,ndImg2)

        # save fused images
        if is_save:
            filename = test_dataset.getName(k)
            resImg.save(os.path.join('results',datasets[j],filename+'.png'), format='PNG', compress_level=0)

    meanArray1 = np.array([en_sum1,sd_sum1,sf_sum1,ag_sum1,vif_sum1,qabf_sum1])/ 1
    meanArray2 = np.array([en_sum2,sd_sum2,sf_sum2,ag_sum2,vif_sum2,qabf_sum2])/ 1
    # meanArray = np.array([en_sum,sd_sum,sf_sum,ag_sum,vif_sum,qabf_sum])/ len(test_dataset)
    drawLineChart(meanArray1,meanArray2)

    # when evluating time costs, remember to stop writing images by setting is_save = False