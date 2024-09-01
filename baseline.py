#!/usr/bin/env python
# coding: utf-8

# In[45]:


import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image
import io
import cv2
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from resnest.torch import resnest101
import torch.optim as optim
import random
from sklearn.metrics import roc_curve, auc


# ### 图像部分的baseline

# In[15]:


work_dir = '/Users/zhouyangfan/Desktop/kaggle/isic/isic-2024-challenge/'
train_hdf5_path = work_dir + 'train-image.hdf5'
train_csv_path = work_dir + 'train-metadata.csv'


# In[30]:


device = torch.device("cuda" if torch.cuda.is_available() else "mps")


# In[7]:


# csv文件读入
train_csv = pd.read_csv(train_csv_path)
train_csv


# In[8]:


train_csv


# In[9]:


# def get_train_file_path(image_id):
#     return f"{work_dir}train-image/image/{image_id}.jpg"
# 样本均衡
train_csv_positive = train_csv[train_csv.target == 1]
train_csv_negative = train_csv[train_csv.target == 0].sample(frac=1.0)
# 正负样本比 1:5
train_csv_balanced = pd.concat([train_csv_positive, train_csv_negative.iloc[:train_csv_positive.shape[0] * 5, :]]).sample(frac=1.0).reset_index(drop=True)
# train_csv_balanced
train_csv_balanced["file_path"] = train_csv_balanced["isic_id"].apply(get_train_file_path)
# 训练集和验证集划分
train_csv_balanced["fold"] = np.random.randint(1, 6, size = train_csv_balanced.shape[0])
valid_csv_fold = train_csv_balanced[train_csv_balanced.fold == 5]
train_csv_fold = train_csv_balanced[train_csv_balanced.fold != 5]


# In[11]:


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
set_seed(88)


# In[129]:


import albumentations as A
from albumentations.pytorch import ToTensorV2

aug_transform = A.Compose([
    A.RandomRotate90(),
    A.Flip(),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.1, p=0.5),
    A.Resize(200, 200),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

base_transform = A.Compose([
    A.Resize(200, 200),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


# In[142]:


class MelanomaDataset(Dataset):
    def __init__(self, csv,  mode, hdf5_data_path, transform=None):
        self.csv = csv.reset_index(drop=True)
        self.mode = mode
        self.hdf5_data = h5py.File(hdf5_data_path, 'r')
        self.transform = transform

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):

        row = self.csv.iloc[index]
        img_name = row['isic_id']
        image = np.array(self.hdf5_data[img_name])
        image = np.array(Image.open(io.BytesIO(image)))#, dtype=np.float32
        #image = cv2.imread(row.file_path)
        
        
        if self.transform is not None:
            image = image.astype(np.float32)
            res = self.transform(image=image)
            image = res['image']
        else:
            image = cv2.resize(image, (224, 224))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32)
            image = image.transpose(2, 0, 1)
        
        data = torch.tensor(image).float()

        if self.mode == 'test':
            return data
        else:
            return data, torch.tensor(self.csv.iloc[index].target).long()


# In[143]:


class CNN(nn.Module):
    def __init__(self, output_size, no_columns):
        super().__init__()
        self.no_columns, self.output_size = no_columns, output_size
        
        # Define Feature part (IMAGE)
        self.features = nn.Sequential(
            nn.Conv2d(self.no_columns, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        
        # Define Classification part
        self.classification = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.output_size),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, image, prints=False):
        
        if prints: print('Input Image shape:', image.shape, '\n')
                         # +
                         # 'Input csv_data shape:', csv_data.shape)
        
        #print("Begin", image.size())
        # Image CNN
        image = self.features(image)
        #print("After conv2:", image.size())
        image = self.avgpool(image)
        image = torch.flatten(image, 1)
        
        if prints: print('Features Image shape:', image.shape)
        
        out = self.classification(image)
        if prints: print('Out shape:', out.shape)
        
        return out


# In[144]:


# class Resnest_Melanoma(nn.Module):
#     def __init__(self, enet_type, out_dim,pretrained = False):
#         super(Resnest_Melanoma, self).__init__()
#         self.enet = resnest101(pretrained=pretrained)
#         in_ch = self.enet.fc.out_features
#         self.myfc = nn.Linear(in_ch, out_dim)
        
#     def extract(self, x):
#         x = self.enet(x)
#         return x

#     def forward(self, x, x_meta=None):
#         x = self.extract(x)
#         return self.myfc(x.squeeze(-1).squeeze(-1))


# In[145]:


# model = resnest101(pretrained=False)
# model.load_state_dict(torch.load(model_path, 'mps'))
# a = torch.load(model_path, 'mps')
# model_path = work_dir + "resnest101-22405ba7.pth"
# import torch

# # 5.加载ResNet101模型
# model = resnest101(pretrained=False)
# # 加载预训练好的ResNet模型
# #model.load_state_dict(torch.load(model_path, 'mps'))
# # # 冻结模型参数
# # for param in model.parameters():
# #     param.requires_grad = False
# model.fc = nn.Linear(2048, 2)
# model.to("mps")
# for i in torch.load(model_path, 'mps'):
#     if 
#     print(i)
#device = torch.device("mps")
#device = torch.device("cpu")
# model_resnet = Resnest_Melanoma('resnest101', out_dim = 2)
# model_resnet.to(device)
# model_resnet.train()
# for (data, target) in train_loader:
#     data, target = data.to(device), target.to(device)
#     logits = model_resnet(data)
#     print(logits)
#     break


# In[ ]:





# In[146]:


criterion = nn.CrossEntropyLoss()
def train_epoch(model, loader, optimizer):

    model.train()
    train_loss = []
    bar = tqdm(loader)
    for (data, target) in bar:
        optimizer.zero_grad()
       
        data, target = data.to(device), target.to(device)
        logits = model(data)

        # print(logits)
        # print(target)
        
        loss = criterion(logits, target)

        loss.backward()
        
        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))

        # 观察是否存在梯度消失
        # param_grad_list = []
        # for param in model.parameters():
        #     param_grad_list.append(param.grad.abs().sum())
        # print(param_grad_list[:2])

    train_loss = np.mean(train_loss)
    return train_loss


# In[147]:


#train_loss = train_epoch(model, train_loader, optimizer)


# In[148]:


def val_epoch(model, loader, n_test=1):

    model.eval()
    val_loss = []
    LOGITS = []
    PROBS = []
    TARGETS = []
    bar = tqdm(loader)
    
    with torch.no_grad():
        for (data, target) in bar:
            
            data, target = data.to(device), target.to(device)
            
            logits = model(data)
            # print(logits)
            # print(target)
            loss = criterion(logits, target)
            probs = logits.softmax(1)

            val_loss.append(loss.detach().cpu().numpy())
            LOGITS.append(logits.detach().cpu())
            PROBS.append(probs.detach().cpu())
            TARGETS.append(target.detach().cpu())
            
   
    # 当前epoch的最后100次损失
    smooth_loss = sum(val_loss[-100:]) / min(len(val_loss), 100)
    bar.set_description('loss: %.5f, smth: %.5f' % (loss, smooth_loss))
    # 当前epoch的的损失
    val_loss = np.mean(val_loss)

    # 计算auc和准确率
    LOGITS = torch.cat(LOGITS).numpy()
    PROBS = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()

    val_auc = roc_auc_score((TARGETS == 1).astype(float), PROBS[:, 1])  
    val_pauc = pauc_cal((TARGETS == 1).astype(float), PROBS[:, 1])
    acc = (PROBS.argmax(1) == TARGETS).mean() * 100.
    
    detection_rate = np.logical_and(PROBS.argmax(1)==1, TARGETS == 1).sum()/(TARGETS == 1).sum()* 100.
    
    return val_loss, val_auc, PROBS, TARGETS, acc, detection_rate, val_pauc


# In[149]:


#@staticmethod
def pauc_cal(y_true, y_scores, tpr_threshold=0.8):
    from sklearn.metrics import roc_curve, auc
        
    # Rescale labels: set 0s to 1s and 1s to 0s (because sklearn only has max_fpr, not min_tpr)
    rescaled_labels = abs(np.asarray(y_true) - 1)

    # Flip the prediction scores to their complements (to work with rescaled label)
    flipped_preds = -1.0 * np.asarray(y_scores)

    # Calculate the maximum false positive rate based on the given TPR threshold
    max_fpr = abs(1 - tpr_threshold)

    # Calculate the ROC curve
    fpr, tpr, _ = roc_curve(rescaled_labels, flipped_preds, sample_weight=None)

    # Find the index where FPR exceeds max_fpr
    interp_idx = np.searchsorted(fpr, max_fpr, 'right')

    # Define points for linear interpolation
    x_interp = [fpr[interp_idx - 1], fpr[interp_idx]]
    y_interp = [tpr[interp_idx - 1], tpr[interp_idx]]

    # Add interpolated point to TPR and FPR arrays
    tpr = np.append(tpr[:interp_idx], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:interp_idx], max_fpr)

    # Calculate the partial AUC
    partial_auc = auc(fpr, tpr)
    
    return partial_auc


# In[157]:


traindf = MelanomaDataset(train_csv_fold, "train",train_hdf5_path,aug_transform)
validdf = MelanomaDataset(valid_csv_fold, "train",train_hdf5_path,base_transform)
train_loader = torch.utils.data.DataLoader(traindf, batch_size=32)
valid_loader = torch.utils.data.DataLoader(validdf, batch_size=32)

model = CNN(output_size=2, no_columns=3)
#model = Resnest_Melanoma('resnest101', out_dim = 2, pretrained=False)

optimizer = optim.Adam(model.parameters(), lr=0.0001)#lr=0.0001
model.to(device)


# In[151]:


# #模型信息打印
# model.parameters()
# for i in model.parameters():
#     print(i.shape)


# In[152]:


# a = torch.tensor([[[1,2,3],[2,3,4]],[[0,0,0],[0,0,0]]])
# a.sum(axis = 2)


# In[153]:


#train_loss = train_epoch(model, train_loader, optimizer)


# In[154]:


#val_loss, val_auc, PROBS, TARGETS, acc,  detection_rate= val_epoch(model, valid_loader, optimizer)


# In[158]:


# 开始训练
train_loss_list = []
val_loss_list = []
val_pauc_list = []

pauc_max = 0.
model_file = work_dir + 'auc_best_model_cnnhdf5.pth'
for epoch in range(1, 30):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_loss, auc, PROBS, TARGETS, acc, detection_rate, pauc = val_epoch(model, valid_loader)
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)
    val_pauc_list.append(pauc)

    if pauc > pauc_max:
        print('pauc_max ({:.6f} --> {:.6f}),acc {:.6f}, detection_rate:{:.6f}, Saving model ...'.format(pauc_max, pauc, acc,detection_rate))
        torch.save(model.state_dict(), model_file)
        pauc_max = pauc


# In[159]:


plt.plot(range(len(val_pauc_list)), val_pauc_list, label="val_pauc_list")
plt.plot(range(len(train_loss_list)), train_loss_list, label="train_loss_list")
plt.plot(range(len(val_loss_list)), val_loss_list, label="val_loss_list")
#plt.plot( range(history.shape[0]), history["Valid AUROC"].values, label="Valid AUROC")
plt.xlabel("epochs")
plt.ylabel("pAUROC")
plt.grid()
plt.legend()
plt.show()


# In[ ]:


## 预测效果的可视化


# In[168]:


import random
random.choice([1,2,3])


# In[167]:





# In[682]:


plt.figure(figsize=(16,3))
j = 1
for i in range(validdf.__len__()):
    if TARGETS[i]:
        plt.subplot(2, 6, j)
        plt.imshow(validdf.__getitem__(i)[0].permute(1, 2, 0)/225)
        plt.axis('off')
        j += 1
    if j>12:
        break


# In[ ]:





# In[ ]:





# In[169]:





# In[170]:


# from torchvision import models
# Inception = models.inception_v3(pretrained = False)
# Resnet50 = models.resnet50(pretrained = True)
# VGG = models.vgg19(pretrained = True)

# Alexnet = models.alexnet(pretrained=True)
# VGG_bn = models.vgg19_bn(pretrained = True)


# In[ ]:


VGG = models.vgg19(pretrained = True)
VGG_bn = models.vgg19_bn(pretrained = True)


# In[ ]:


Alexnet = models.alexnet(pretrained=True)


# In[ ]:




