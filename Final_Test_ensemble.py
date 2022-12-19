

from PIL import Image
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import transforms, datasets
import torch.nn.functional as F 
import os
import shutil
from timm import create_model
from sklearn.metrics import confusion_matrix,recall_score,precision_score,accuracy_score,f1_score
from sklearn.metrics import precision_recall_fscore_support
import warnings
from collections import OrderedDict
warnings.filterwarnings('ignore')
import math
import pandas as pd
import numpy as np
import time
from tqdm import tqdm, trange
import json
import os
import random

class padding_resize:
    def __init__(self, desired_size):
        try:
            if(len(desired_size)==2):
                desired_size=desired_size[0]
        except :
            desired_size=desired_size
        self.desired_size = desired_size

    def __call__(self, im):
        #angle = random.choice(self.angles)        
        desired_size = self.desired_size
        old_size = im.size
        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        im = im.resize(new_size, Image.BICUBIC)
        new_im = Image.new("RGB", (desired_size, desired_size))
        new_im.paste(im, ((desired_size-new_size[0])//2,(desired_size-new_size[1])//2))

        return new_im
class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def Get_Model(model_name,pkl_path,num_classes=33):
    model = create_model(model_name, pretrained=False,num_classes=num_classes, checkpoint_path=pkl_path)
    return model

if __name__ == '__main__':
    
    
    with open("./class_to_idx.json") as f:
        class_to_idx = json.load(f)
    idx_to_class={}
    for key,value in class_to_idx.items():
        idx_to_class[value] = key
    
    pp = ['public','private']
    mod =  pp[0]
    
    template = pd.read_csv('./submission_example.csv')
    template.loc[:,'label']=-1
    
    #identified_template = pd.read_csv('./submission_template.csv')
    #identified = identified_template.loc[template['category']!=-1].loc[:,'filename'].tolist()
    
    #val_path = r"C:\Users\chen_hung\Desktop\AI_CUP_2022\Datasets\training\val"
    
    farming_test_set = "./public_test"
    
    #******************************************************************************************************************************
    BATCH_SIZE = 8
    Num_workers = 2
    
    resize = 384
    #mean_384_one_fold = [0.5075337404927281 ,0.45864544276917535 ,0.4169235386212412] 
    #std_384_one_fold = [0.2125643051799512 ,0.2385082849964861 ,0.22386483801695406]
    #mean_384_two_fold = [0.45625534556274666, 0.4220624936173144, 0.3649616601198825]
    #std_384_two_fold = [0.2143212828816861, 0.2210437745632625, 0.2062174242104951]
    
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    
    mean = [0.45925, 0.48785, 0.42035]#(0.5,0.5,0.5)
    std = [0.25080, 0.24715, 0.29270]#(0.5,0.5,0.5)
    
    tfm = transforms.Compose([
        transforms.Resize(resize, interpolation=Image.BICUBIC),
        transforms.CenterCrop(resize),
        transforms.ToTensor(),
    ])
    

    
    tfm_1 = transforms.Compose([
        transforms.Normalize(mean, std)
    ])
    
    mean = (0.5,0.5,0.5)
    std = (0.5,0.5,0.5)
    
    tfm_2 = transforms.Compose([
        transforms.Normalize(mean, std)
    ])
    
    #resize = 256
    # tfm_pd = transforms.Compose([
    #     transforms.Resize(resize, interpolation=Image.BICUBIC),
    #     transforms.CenterCrop(resize),
    #     #padding_resize(resize),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean, std)
    # ])
    
    #tfm_one = transforms.Normalize(mean_384_one_fold, std_384_one_fold)
    #tfm_two = transforms.Normalize(mean_384_two_fold, std_384_two_fold)
    
    farming_test_path = ImageFolderWithPaths("./total_test",transform=tfm)
    farming_test_Loader_path = torch.utils.data.DataLoader(dataset=farming_test_path,batch_size=BATCH_SIZE,shuffle=False,num_workers=Num_workers)
    
    #pri_farming_test_path = ImageFolderWithPaths("./private_test",transform=tfm)
    #pri_farming_test_Loader_path = torch.utils.data.DataLoader(dataset=farming_test_path,batch_size=BATCH_SIZE,shuffle=False,num_workers=Num_workers)
    
    #farming_pd_test_path = ImageFolderWithPaths(farming_test_set,transform=tfm_pd)
    #farming_pd_test_Loader_path = torch.utils.data.DataLoader(dataset=farming_pd_test_path,batch_size=BATCH_SIZE,shuffle=False,num_workers=Num_workers)
    
    #val_data = datasets.ImageFolder(val_path)
    #print(val_data.classes)
    #print(orchid_test_path.calsses)
    #*************************************************************************************************************************************************
    
    e_Swin_pkl = "./weight/swinv2_large_newA_mean_std_distillation_rotate_checkpoint-28.pth.tar"#None#"./pytorch-image-models/output/0__swin_large__official_95.2887/model_best.pth.tar"
    e_Swin_model = Get_Model('swinv2_large_window12to24_192to384_22kft1k',pkl_path=e_Swin_pkl)
    e_Swin_model = nn.DataParallel(e_Swin_model)
    e_Swin_model = e_Swin_model.cuda()
    
    e2_Swin_pkl = "./weight/swinv2_large_newB_mean_std_distillation_rotate_checkpoint-30.pth.tar"#None#"./pytorch-image-models/output/0__swin_large__official_95.2887/model_best.pth.tar"
    e2_Swin_model = Get_Model('swinv2_large_window12to24_192to384_22kft1k',pkl_path=e2_Swin_pkl)
    e2_Swin_model = nn.DataParallel(e2_Swin_model)
    e2_Swin_model = e2_Swin_model.cuda()
    
    Swin_pkl = "./weight/A_swinv2_large_384_mean_std_ckt38.pth.tar"#None#"./pytorch-image-models/output/0__swin_large__official_95.2887/model_best.pth.tar"
    Swin_model = Get_Model('swinv2_large_window12to24_192to384_22kft1k',pkl_path=Swin_pkl)
    Swin_model = nn.DataParallel(Swin_model)
    Swin_model = Swin_model.cuda()
    
    #"./weight/B_swinv2_large_384_mean_std_rotate_ckt32.pth.tar"
    Swin_pkl_2 = "./weight/B_swinv2_large_384_mean_std_ckt26_server4.pth.tar"
    #'./weight/B_swinv2_large_384_mean_std_ckt24.pth.tar'
    #"./weight/B_swinv2_large_384_mean_std_ckt16_server4.pth.tar"#None#"./pytorch-image-models/output/0__swin_large__official_95.2887/model_best.pth.tar"
    Swin_model_2 = Get_Model('swinv2_large_window12to24_192to384_22kft1k',pkl_path=Swin_pkl_2)
    Swin_model_2 = nn.DataParallel(Swin_model_2)
    Swin_model_2 = Swin_model_2.cuda()
    
    #name = 'A_B_swinv2_38_9_old_beit_16_14_ensemble'#os.path.split(Swin_pkl)[1].split('.pth.tar')[0]
    name = 'A_B_swinv2_38_26_old_beit_16_14_da_28_30_ensemble_final'
    
    Swin_pkl_3 = "./weight/newA_beit_ckt_16.pth.tar"#None#"./pytorch-image-models/output/0__swin_large__official_95.2887/model_best.pth.tar"
    Swin_model_3 = Get_Model('beit_large_patch16_384',pkl_path=Swin_pkl_3)
    Swin_model_3 = nn.DataParallel(Swin_model_3)
    Swin_model_3 = Swin_model_3.cuda()
    

    Swin_pkl_4 = "./weight/B_beit_ckt_14.pth.tar"#None#"./pytorch-image-models/output/0__swin_large__official_95.2887/model_best.pth.tar"
    Swin_model_4 = Get_Model('beit_large_patch16_384',pkl_path=Swin_pkl_4)
    Swin_model_4 = nn.DataParallel(Swin_model_4)
    Swin_model_4 = Swin_model_4.cuda()
    
    #name = 'newA_swinv2_30_16_beit_ensemble'#os.path.split(Swin_pkl)[1].split('.pth.tar')[0]
    
    times = 0
    
    print(name)
    progress_1 = tqdm(total=len(farming_test_path))
    #progress_2 = tqdm(total=len(pri_farming_test_path))
    #*************************************************************************************************************************************************
    final_outputs = []
    not_ok = []
    ok_count = 0
    error_data = []
    with torch.no_grad():
            Swin_model.eval()
            Swin_model_2.eval()
            Swin_model_3.eval()
            Swin_model_4.eval()
            e_Swin_model.eval()
            e2_Swin_model.eval()
            
            for step, (batch_x,label_y,path) in enumerate(farming_test_Loader_path):
                
                img_file = np.array([os.path.split(p)[1] for p in path])
                #print(path)
                
                test1 = Variable(tfm_1(batch_x)).cuda()
                output_1 = Swin_model(test1)
                #test2 = Variable(tfm_1(batch_x)).cuda()
                output_2 = Swin_model_2(test1)
                output_e = e_Swin_model(test1)
                output_e2 = e2_Swin_model(test1)
                
                test3 = Variable(tfm_2(batch_x)).cuda()
                output_3 = Swin_model_3(test3)
                output_4 = Swin_model_4(test3)
                
                s_out = F.softmax(output_1, dim=1)+F.softmax(output_2, dim=1)+F.softmax(output_3, dim=1)+F.softmax(output_4, dim=1)+F.softmax(output_e, dim=1)+F.softmax(output_e2, dim=1)
                
                
                ans = torch.max(s_out,1)[1].squeeze()
                final_ans = ans.cpu().data.numpy()
                
                #final_ans = [random.randint(0,10)for i in range(len(img_file))]
                #print(final_ans)
                #print("{}/{},{}".format(step,len(orchid_test_Loader_path),img_file))
                
                for img_file,final_ans in zip(img_file,final_ans):
                    try:
                            #t1 = time.time()
                            template.loc[template['filename']==img_file,'label']=idx_to_class[final_ans]
                            #t2 = time.time()
                            #print(t2-t1)
                            ok_count+=1
                    except:
                        #print('error -->img:{},ans:{}'.format())
                        error_data.append([img_file,final_ans])
                    progress_1.update(1)
            # for step, (batch_x,label_y,path) in enumerate(pri_farming_test_Loader_path):
                
            #     img_file = np.array([os.path.split(p)[1] for p in path])
            #     #print(path)
            #     test1 = Variable(tfm_1(batch_x)).cuda()
            #     output_1 = Swin_model(test1)
            #     #test2 = Variable(tfm_1(batch_x)).cuda()
            #     output_2 = Swin_model_2(test1)
            #     output_e = e_Swin_model(test1)
            #     output_e2 = e2_Swin_model(test1)
                
            #     test3 = Variable(tfm_2(batch_x)).cuda()
            #     output_3 = Swin_model_3(test3)
            #     output_4 = Swin_model_4(test3)
                
            #     s_out = F.softmax(output_1, dim=1)+F.softmax(output_2, dim=1)+F.softmax(output_3, dim=1)+F.softmax(output_4, dim=1)+F.softmax(output_e, dim=1)+F.softmax(output_e2, dim=1)
                
                
            #     ans = torch.max(s_out,1)[1].squeeze()
            #     final_ans = ans.cpu().data.numpy()
            #     #print(final_ans)
            #     #print("{}/{},{}".format(step,len(orchid_test_Loader_path),img_file))
                
            #     for img_file,final_ans in zip(img_file,final_ans):
            #         try:
            #                 #t1 = time.time()
            #                 template.loc[template['filename']==img_file,'label']=idx_to_class[final_ans]
            #                 #t2 = time.time()
            #                 #print(t2-t1)
            #                 ok_count+=1
            #         except:
            #             #print('error -->img:{},ans:{}'.format())
            #             error_data.append([img_file,final_ans])
            #         progress_2.update(1)
            
                #test_pd = Variable(test_pd.unsqueeze(0)).cuda()
                #output_2 = Swin_model_2(test_pd)
                
                
                #final_outputs = F.softmax(output_1, dim=1)+F.softmax(output_2, dim=1)
                
                #+F.softmax(outpust_1_one, dim=1)+F.softmax(outpust_2_one, dim=1)
                
                #final_outputs = F.softmax(outpust_0_one, dim=1)+F.softmax(outpust_1_one, dim=1)+F.softmax(outpust_2_one, dim=1)+
                #            F.softmax(outpust_0_two, dim=1)+F.softmax(outpust_1_two, dim=1)+F.softmax(outpust_2_two, dim=1)
                            
                #ans = torch.max(final_outputs,1)[1].squeeze()
                #final_ans = ans.cpu().data.numpy()
                #print(final_ans)
                #print("{}/{},{}".format(step,len(orchid_test_Loader_path),img_file))
                
                
                # try:
                #     #t1 = time.time()
                #     template.loc[template['filename']==img_file,'label']=idx_to_class[final_ans]
                #     #t2 = time.time()
                #     #print(t2-t1)
                #     ok_count+=1
                # except:
                #         #print('error -->img:{},ans:{}'.format())
                #     error_data.append([img_file,final_ans])
                #progress.update(1)
                #     print("{}/{} is not ok".format(label_y,img_file) )
                #     not_ok.append([label_y,img_file])
                # if(step%10000==0):
                #     template.to_csv("./Ans/{}_outpust_interrupt_{}.csv".format(name,step), index=False)
                #     print("{}/{},{}".format(step,len(orchid_test_Loader_path),img_file))
                
            #print("{}/{}".format(ok_count,step))
    if(len(error_data)!=0):
        for i in error_data:
            print(i)
    

    
    template.to_csv("./Ans/{}_outpust.csv".format(name), index=False)
    
    # for i in not_ok:
    #     t_fi= open('/Ans/Not_ok.txt'.format(epoch+1),'w')
    #     t_fi.writelines("{}/{} \n".format(i[0],i[1]))
    #     t_fi.close()
    #img_file = "dmlg79vsu0.jpg"
    #model_ans = 1
    #template.loc[template['filename']==img_file,'category']=model_ans
    
    
    
    
    
    