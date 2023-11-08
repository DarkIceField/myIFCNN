import os
import time
import torch
from IFCNNmodel import myIFCNN as IFCNN
from model import myIFCNN as myModel
os.environ['CUDA_VISIBLE_DEVICES']='3'

from torchvision import transforms
from torch.autograd import Variable

from PIL import Image
import numpy as np
import pandas as pd
from dataset import ImgEvalSet
from evaluator import Evaluator
import matplotlib.pyplot as plt
from draw import drawLineChart

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))
# load pretrained model


datasets = [ 'MSRS'] # Color MultiFocus, Infrared-Visual, MeDical image datasets
# datasets = [ 'MSRS','TNO_21','RoadScene', 'MRI-PET','MRI_SPECT'] # Color MultiFocus, Infrared-Visual, MeDical image datasets
dIndex = 0
is_save = True             # if you do not want to save images, then change its value to False
modelName = 'IFCNN'
testRoot = '../fusion_datasets/test/'

for j in range(len(datasets)):
    begin_time = time.time()
    setRoot = os.path.join(testRoot, datasets[j])
    resRoot = os.path.join('results',modelName,datasets[j])
    csvFile = os.path.join(resRoot,'evalResult.csv')
    if os.path.exists(csvFile):
        cData =pd.read_csv(csvFile)
    else:
        cData = pd.DataFrame(index=[],
                       columns=['file_name',
                                'EN',
                                'SD',
                                'SF',
                                'AG',
                                'VIFF',
                                'Qabf'
                        ])
        
    evl_dataset = ImgEvalSet(setRoot,resRoot)
    # for k in range(1):
    en_sum,sd_sum,sf_sum,ag_sum,vif_sum,qabf_sum=0.0,0.0,0.0,0.0,0.0,0.0
    for k in range(len(evl_dataset)):
        img1, img2, resImg = evl_dataset[k]
        fileName = evl_dataset.getName(k)
        print(fileName)
        # evaluate Image
        ev = Evaluator()
        ev.input_check(img1)
        ev.input_check(img2)
        ev.input_check(resImg)
        en_val=  ev.EN(resImg)
        sd_val=  ev.SD(resImg)
        sf_val=  ev.SF(resImg)
        ag_val=  ev.AG(resImg)
        vif_val=  ev.VIFF(resImg,img1,img2)
        qabf_val=  ev.Qabf(resImg,img1,img2)
        en_sum+=  en_val
        sd_sum+=  sd_val
        sf_sum+=  sf_val
        ag_sum+=  ag_val
        vif_sum+=  vif_val
        qabf_sum+=  qabf_val
        newRow = pd.Series([
                fileName,
                en_val,
                sd_val,
                sf_val,
                ag_val,
                vif_val,
                qabf_val
            ], index=['file_name','EN','SD','SF','AG','VIFF','Qabf'])
        cData = cData._append(newRow, ignore_index=True)

    xlist = ['EN','SD','SF','AG','VIFF','Qabf']
    meanArray = np.array([en_sum,sd_sum,sf_sum,ag_sum,vif_sum,qabf_sum])/ len(evl_dataset)
    newRow = pd.Series(meanArray.tolist().insert(0,'mean'), index=['file_name','EN','SD','SF','AG','VIFF','Qabf'])
    cData = cData._append(newRow, ignore_index=True)
    cData.to_csv(csvFile, index=False)
    drawLineChart(xlist,None,'metrics',meanArray)

    # when evluating time costs, remember to stop writing images by setting is_save = False
