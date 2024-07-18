#!/usr/bin/env python
# coding: utf-8

# In[350]:


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import cv2
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from sklearn.metrics import roc_auc_score


# ### 图像部分的baseline

# In[13]:


work_dir = '/Users/zhouyangfan/Desktop/kaggle/isic/isic-2024-challenge/'
train_hdf5 = work_dir + 'train-image.hdf5'
test_hdf5 = work_dir + 'test-image.hdf5'
train_csv_path = work_dir + 'train-metadata.csv'
test_csv_path = work_dir + 'test-metadata.csv'


# In[7]:


# csv文件读入
train_csv = pd.read_csv(train_csv_path)
train_csv
test_csv = pd.read_csv(test_csv_path)
test_csv


# In[14]:


def get_train_file_path(image_id):
    return f"{work_dir}train-image/image/{image_id}.jpg"


# In[ ]:


# # hdf5文件读入
# train_dataset = h5py.File(train_hdf5, 'r')
# train_images = {}
# for image in train_dataset.keys():
#     train_image = train_dataset[image]
#     img_plt = Image.open(io.BytesIO(np.array(train_image)))
#     img_array = np.array(img_plt)
#     train_images[image] = img_array
# test_dataset = h5py.File(test_hdf5, 'r')
# test_images = {}
# for image in test_dataset.keys():
#     test_image = test_dataset[image]
#     img_plt = Image.open(io.BytesIO(np.array(test_image)))
#     img_array = np.array(img_plt)
#     test_images[image] = img_array


# In[123]:


# 样本均衡
train_csv_positive = train_csv[train_csv.target == 1]
train_csv_negative = train_csv[train_csv.target == 0].sample(frac=1.0)
# 正负样本比 1:5
train_csv_balanced = pd.concat([train_csv_positive, train_csv_negative.iloc[:train_csv_positive.shape[0] * 5, :]]).sample(frac=1.0).reset_index(drop=True)


# In[323]:


train_csv_negative.shape


# In[322]:


train_csv_balanced.shape


# In[124]:


train_csv_balanced


# In[127]:


# train_csv_balanced
train_csv_balanced["file_path"] = train_csv_balanced["isic_id"].apply(get_train_file_path)


# In[128]:


# 训练集和验证集划分
train_csv_balanced["fold"] = np.random.randint(1, 6, size = train_csv_balanced.shape[0])
valid_csv_fold = train_csv_balanced[train_csv_balanced.fold == 5]
train_csv_fold = train_csv_balanced[train_csv_balanced.fold != 5]


# In[330]:


train_csv_fold.shape


# In[129]:


# 不加 transform
class MelanomaDataset(Dataset):
    def __init__(self, csv,  mode, transform=None):
        self.csv = csv.reset_index(drop=True)
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):

        row = self.csv.iloc[index]
        #image = self.img_dict[row.isic_id]
        image = cv2.imread(row.file_path)
        image = cv2.resize(image, (200, 200))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)
        
        data = torch.tensor(image).float()

        if self.mode == 'test':
            return data
        else:
            return data, torch.tensor(self.csv.iloc[index].target).long()


# In[310]:


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
        
        # Image CNN
        image = self.features(image)
        #print("After conv2:", image.size())
        image = self.avgpool(image)
        image = torch.flatten(image, 1)
        
        if prints: print('Features Image shape:', image.shape)
        
        out = self.classification(image)
        if prints: print('Out shape:', out.shape)
        
        return out


# In[400]:


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


        # param_grad_list = []
        # for param in model.parameters():
        #     param_grad_list.append(param.grad.abs().sum())
        # print(param_grad_list)

    train_loss = np.mean(train_loss)
    return train_loss
    
def val_epoch(model, loader, n_test=1):

    model.eval()
    val_loss = []
    LOGITS = []
    PROBS = []
    TARGETS = []
    
    with torch.no_grad():
        for (data, target) in tqdm(loader):
            
            data, target = data.to(device), target.to(device)
            
            logits = model(data)
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
    return val_loss, val_auc


# In[401]:


torch.cat([logits.softmax(1),logits.softmax(1)])


# In[402]:


target


# In[403]:


valid_csv_fold.shape


# In[431]:





# In[418]:


# a = torch.tensor([[[1,2,3],[2,3,4]],[[0,0,0],[0,0,0]]])
# a.sum(axis = 2)


# In[419]:


#train_loss = train_epoch(model, train_loader, optimizer)


# In[420]:


#valid_loss = val_epoch(model, valid_loader, optimizer)


# In[432]:


traindf = MelanomaDataset(train_csv_fold, "train")
validdf = MelanomaDataset(valid_csv_fold, "train")
train_loader = torch.utils.data.DataLoader(traindf, batch_size=32)
valid_loader = torch.utils.data.DataLoader(validdf, batch_size=32)

model = CNN(output_size=2, no_columns=3)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

device = torch.device("mps")
model.to(device)

# 开始训练
train_loss_list = []
val_loss_list = []
val_auc_list = []

auc_max = 0.
model_file = work_dir + 'auc_best_model.pth'
for epoch in range(1, 11):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_loss, auc = val_epoch(model, valid_loader)
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)
    val_auc_list.append(auc)

    if auc > auc_max:
        print('auc_max ({:.6f} --> {:.6f}). Saving model ...'.format(auc_max, auc))
        torch.save(model.state_dict(), model_file)
        auc_max = auc


# In[ ]:


train_loss_list


# In[423]:


val_loss_list


# In[424]:


val_auc_list


# In[433]:


plt.plot(range(len(val_auc_list)), val_auc_list, label="val_auc_list")
plt.plot(range(len(train_loss_list)), train_loss_list, label="train_loss_list")
plt.plot(range(len(val_loss_list)), val_loss_list, label="val_loss_list")
#plt.plot( range(history.shape[0]), history["Valid AUROC"].values, label="Valid AUROC")
plt.xlabel("epochs")
plt.ylabel("AUROC")
plt.grid()
plt.legend()
plt.show()


# In[447]:


stack = {}
if stack:
    print("yes")
    print(stack)
if not stack:
    print("no")
    print(stack.add("1"))
    print(stack)


# In[ ]:




