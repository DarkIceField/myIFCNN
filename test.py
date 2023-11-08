import os
import time
import torch
from IFCNNmodel import myIFCNN as IFCNN
from model import myIFCNN
os.environ['CUDA_VISIBLE_DEVICES']='3'

from torchvision import transforms
from torch.autograd import Variable

from PIL import Image
import numpy as np
import pandas as pd
from dataset import ImgFusionSet
from evaluator import Evaluator
import matplotlib.pyplot as plt
from draw import drawLineChart
import cv2 as cv
from tqdm import tqdm
from loss import ir_loss, vi_loss,ssim_loss,gra_loss

def validate(model,dataloader,device,weight):
    print('start test:')
    lossSum,loss_irSum,loss_viSum,loss_ssimSum,loss_graSum=0.0,0.0,0.0,0.0,0.0
    with torch.no_grad():
        for index, (ir, vi) in tqdm(enumerate(dataloader), total=len(dataloader)):
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

    loss_mean=lossSum/len(dataloader)
    loss_ir_mean=loss_irSum/len(dataloader)
    loss_vi_mean=loss_viSum/len(dataloader)
    loss_ssim_mean=loss_ssimSum/len(dataloader)
    loss_gra_mean=loss_graSum/len(dataloader)
    print('end test,loss_ir:{:.4f},loss_vi:{:.4f},loss_ssim:{:.4f},loss_gra:{:.4f},loss_total:{:.4f}'.format(loss_ir_mean,loss_vi_mean,loss_ssim_mean,loss_gra_mean,loss_mean))
    return loss_mean

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    # load pretrained model


    IFCNNmodel = IFCNN().to(device)
    mymodel = myIFCNN().to(device=device)
    modelName = 'IFCNN'
    weightFile = os.path.join('checkPoints'+modelName,'model_epoch60.tar')
    trained_weights = torch.load(weightFile)['model_state_dict']
    IFCNNmodel.load_state_dict(trained_weights)
    IFCNNmodel.eval()

    weightFile = os.path.join('checkPoints','model_epoch190.tar')
    # myModel_trained_weights =torch.load(weightFile)['model_state_dict']
    # model_weights = mymodel.state_dict().copy()
    # new_weights ={}

    # for i in model_weights.keys():
    #     if i in trained_weights:
    #         print(i)
    #         new_weights[i]=trained_weights[i]
    #     else:
    #         new_weights[i] = myModel_trained_weights[i]


    # mymodel.load_state_dict(new_weights)
    mymodel.load_state_dict(torch.load(weightFile)['model_state_dict'])
    mymodel.eval()

    totalParam = sum([param.nelement() for param in IFCNNmodel.parameters()])
    print('Total parameters number of model: {:.4f}M'.format(totalParam/1000000))
    insertCol = []

    datasets = [ 'MSRS','RoadScene','TNO_21','TNO_25'] # Color MultiFocus, Infrared-Visual, MeDical image datasets
    # datasets = [ 'MSRS','TNO_21','RoadScene', 'MRI-PET','MRI_SPECT'] # Color MultiFocus, Infrared-Visual, MeDical image datasets
    is_save = False     # if you do not want to save images, then change its value to False
    testRoot = '../fusion_datasets/test/'

    for j in range(len(datasets)):
        print(datasets[j])
        setRoot = os.path.join(testRoot, datasets[j])
        test_dataset = ImgFusionSet(img_dir=setRoot)
        # for k in range(1):
        for k in range(len(test_dataset)):
            img1, img2 = test_dataset[k]
            img1.unsqueeze_(0)
            img2.unsqueeze_(0)
            print(test_dataset.getName(k))
            # perform image fusion
            with torch.no_grad():
                res = IFCNNmodel(Variable(img1.cuda()), Variable(img2.cuda())).squeeze()
                ndRes =res.squeeze().detach().cpu().numpy()
                ndRes = (ndRes-np.min(ndRes))/(np.max(ndRes)-np.min(ndRes)+1e-8)
                ndRes = (ndRes*255).astype(np.uint8)
                # cv.imshow('a',ndRes)
                # cv.waitKey(20000)
                
            # save fused images
            if is_save:
                filename = test_dataset.getName(k)
                dirPath = os.path.join('results',modelName,datasets[j],)
                if not os.path.exists(dirPath):
                    os.mkdir(dirPath)
                cv.imwrite(os.path.join(dirPath,filename+'.png'),ndRes)

        # xlist = ['EN','SD','SF','AG','VIFF','Qabf']
        # meanArray = np.array([en_sum,sd_sum,sf_sum,ag_sum,vif_sum,qabf_sum])/ len(test_dataset)
        # drawLineChart(xlist,meanArray)

        # when evluating time costs, remember to stop writing images by setting is_save = False
        # mean_proc_time = (time.time() - begin_time)/len(test_dataset)
        # print('Mean processing time of {} dataset: {:.3}s'.format(datasets[j], mean_proc_time))
        # insertCol.append(totalParam)
        # insertCol.append(mean_proc_time)
