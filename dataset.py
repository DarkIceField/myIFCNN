from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import torch
import numpy as np

class ImgFusionSet(Dataset):
    def __init__(self,img_dir,transvi = None,transir=None):
        vi_img_dir = os.path.join(img_dir,"vi")
        ir_img_dir = os.path.join(img_dir,"ir")
        self.vi_img_pathes = [os.path.join(vi_img_dir,imgPath) for imgPath in os.listdir(vi_img_dir)]
        self.ir_img_pathes = [os.path.join(ir_img_dir,imgPath) for imgPath in os.listdir(ir_img_dir)]
        self.viTransform = transvi
        self.irTransform = transir
        # print(self.img_pathes)

    def __getitem__(self, item):
        vi_image_path = self.vi_img_pathes[item]
        ir_image_path = self.ir_img_pathes[item]
        vi_image = Image.open(vi_image_path).convert('L')
        ir_image = Image.open(ir_image_path).convert('L')
        if self.viTransform is None:
            transvi = transforms.ToTensor()
        else:
            transvi = self.viTransform
        if self.irTransform is None:
            transir = transforms.ToTensor()
        else:
            transir = self.irTransform

        cropSeed = torch.random.seed()
        torch.random.manual_seed(cropSeed)
        vi_image = transvi(vi_image)
        torch.random.manual_seed(cropSeed)
        ir_image = transir(ir_image)
        return vi_image,ir_image

    def __len__(self):
        return len(self.vi_img_pathes)
    
    def getName(self,item):
        vi_image_path = self.vi_img_pathes[item]
        _,rawFilename = os.path.split(vi_image_path) 
        fileName,_ = os.path.splitext(rawFilename)
        return fileName
    
class ImgEvalSet(Dataset):
    def __init__(self,src_img_dir,res_img_dir):
        vi_img_dir = os.path.join(src_img_dir,"vi")
        ir_img_dir = os.path.join(src_img_dir,"ir")
        self.vi_img_pathes = [os.path.join(vi_img_dir,imgPath) for imgPath in os.listdir(vi_img_dir)]
        self.ir_img_pathes = [os.path.join(ir_img_dir,imgPath) for imgPath in os.listdir(ir_img_dir)]
        self.res_img_pathes = [os.path.join(res_img_dir,imgPath) for imgPath in os.listdir(res_img_dir)]
        # print(self.img_pathes)

    def __getitem__(self, item):
        vi_image_path = self.vi_img_pathes[item]
        ir_image_path = self.ir_img_pathes[item]
        res_image_path = self.res_img_pathes[item]
        vi_image = Image.open(vi_image_path).convert('L')
        ir_image = Image.open(ir_image_path).convert('L')
        res_image = Image.open(res_image_path).convert('L')

        vi_dimage = np.array(vi_image,dtype=np.float_)
        ir_dimage = np.array(ir_image,dtype=np.float_)
        res_dimage = np.array(res_image,dtype=np.float_)

        return vi_dimage,ir_dimage,res_dimage

    def __len__(self):
        return len(self.vi_img_pathes)
    
    def getName(self,item):
        vi_image_path = self.vi_img_pathes[item]
        _,rawFilename = os.path.split(vi_image_path) 
        fileName,_ = os.path.splitext(rawFilename)
        return fileName