import os
import torch
import cv2
import numpy as np
import math
import time
#import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset
from PIL import Image
from torchvision import transforms
import random



class farming_datasets(Dataset): 
    def __init__(self,root,datatxt, target_crop=False,transform=None): 
        self.root = root
        self.target_crop = target_crop
        self.transform = transform
        
        fh = open(datatxt, 'r',encoding="utf_8_sig") 
        data = []
        for line in fh :
            line = line.rstrip()
            words = line.split()
            data.append((words[0],words[1],words[2],words[3]))#圖片，label，座標
            
        self.data = data

    def __getitem__(self, index):
        
            img_file, label , target_x , target_y = self.data[index] 
            img = Image.open(os.path.join(self.root,img_file))

            if self.target_crop :
                img = self.target_crop_fn(img,int(target_x),int(target_y)) 
            if self.transform is not None:
                img = self.transform(img)
            
            return img,int(label)

    def __len__(self): 
        return len(self.data)
    
    def target_crop_fn(self,img,target_x,target_y):

        img_w,img_h = img.size
        
        center_x = math.ceil(img_w/2)
        center_y = math.ceil(img_h/2)
        
        new_center_x = center_x+target_x
        new_center_y = center_y-target_y
        
        left_dis = abs(new_center_x-0)
        upper_dis = abs(new_center_y-0)
        right_dis = abs(img_w-new_center_x)
        bottom_dis = abs(img_h-new_center_y)
    
        min_val = min(left_dis,upper_dis,right_dis,bottom_dis)
        #scale_factory = math.ceil(min_val/img_size)
        
        new_crop_dis = min_val#math.ceil(scale_factory*img_size/2)
        left = abs(new_center_x-new_crop_dis)
        upper = abs(new_center_y-new_crop_dis)
        right = abs(new_center_x+new_crop_dis)
        bottom = abs(new_center_y+new_crop_dis)
        
        new_img = img.crop((left, upper, right, bottom))
        
        return new_img
    
if __name__ == "__main__":
    
    trc = transforms.Compose([ transforms.Resize((288,288)),
                               transforms.ToTensor()])
    tensor_to_Pil = transforms.ToPILImage()
    
    test_data = farming_datasets('./','test.txt',True,trc)
    test_dataloader = DataLoader(test_data,batch_size=2,shuffle=False,num_workers=0)
    

    
    for step,(img,label) in enumerate(test_dataloader):
        print(label)
        pass
    
    pass