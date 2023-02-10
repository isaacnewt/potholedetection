#!/usr/bin/env python
# coding: utf-8

# ## Objective
# 
# This module is prepared with pycharm and will help us detectpot holes for our object detection project.
# 1. It is recommended to be run on google colab.
# 2. Make sure to add the appropriate path names to read the image names and annotation files if you are running locally.
# 3. You can change the runtime to GPU for quick results.
# 
# 
# **Steps to Implement Faster RCNN**
# 
# 1. Import requires libraries and load data
# 2. Data Preprocessing
# 3. Defining model architecture
# 4. Testing the model

# **Import Libraries**

import os
import numpy as np
import pandas as pd
import torch

from PIL import Image
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# unzip the data
get_ipython().system("unzip 'train.zip'")
get_ipython().system("unzip 'test.zip'")

# **Data Preprocessing**

# reading csv file
cdata = pd.read_csv('train/labels.csv')
# cdata.head()

cdata['LabelName'] = cdata['LabelName'].replace({'pothole': 1})

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Let's definine the required transformations
from torchvision import transforms as TF

transform = TF.Compose([
    TF.Resize((224, 224)),
    TF.ToTensor()
])

# defining class to load data
class PotHoleDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None,train=True):
        self.root = root
        self.transforms = transforms
        # load all image files
        self.imgs = os.listdir(root)
        if '.ipynb_checkpoints' in self.imgs:
          self.imgs.remove('.ipynb_checkpoints')
        self.train = train

    def __getitem__(self, idx):
        # load images

        img_name = self.imgs[idx]
        img_path = os.path.join(self.root, img_name)
        img = Image.open(img_path)
        if self.train == False:
            if self.transforms is not None:
                img = self.transforms(img)
            return img,img_name
        else:
            h, w = np.array(img).shape[:2]
            num_objs = cdata[cdata['ImageID'] == img_name].shape[0]
            boxes = []
            cell_type = []
            for i in range(num_objs):
                xmin = cdata[cdata['ImageID'] == img_name]['XMin'].iloc[i]
                xmax = cdata[cdata['ImageID'] == img_name]['XMax'].iloc[i]
                ymin = cdata[cdata['ImageID'] == img_name]['YMin'].iloc[i]
                ymax = cdata[cdata['ImageID'] == img_name]['YMax'].iloc[i]
                target = cdata[cdata['ImageID'] == img_name]['LabelName'].iloc[i]
                boxes.append([xmin, ymin, xmax, ymax])
                cell_type.append(target)
            # convert everything into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(cell_type, dtype=torch.int64)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            #print(boxes)
            if self.transforms is not None:
                img = self.transforms(img)
                boxes[:, 0] = boxes[:, 0] * (224/w)
                boxes[:, 2] = boxes[:, 2] * (224/w)
                boxes[:, 1] = boxes[:, 1] * (224/h)
                boxes[:, 3] = boxes[:, 3] * (224/h)

            return img, target

    def __len__(self):
        return len(self.imgs)

dataset = PotHoleDataset('train/images/', transforms=transform)

# Let's convert list of list to tuple
def collate_fn(batch):
    return tuple(zip(*batch))

data_loader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=8, 
    shuffle=False, 
    collate_fn=collate_fn
)

# testing for one iteration
for batch_x, batch_y in data_loader:
    break

#batch_x[0].shape

plt.imshow(np.transpose(batch_x[1], (1, 2, 0)))

# We need to use the Matplolib patches here
import matplotlib.patches as pc

# plot bounding box
plt.axes()
plt.imshow(np.transpose(batch_x[0], (1, 2, 0)))
for i in range(len(batch_y[0]['boxes'])):
    bbox = batch_y[0]['boxes'][i]
    x1, y1 = bbox[0], bbox[1]
    x2, y2 = bbox[2], bbox[3]
    if batch_y[0]['labels'][i] == 1:
        color = 'red'

    rectangle = pc.Rectangle((x1,y1), x2-x1, y2-y1, fc='none',ec=color)
    plt.gca().add_patch(rectangle)
plt.show()


# **Defining the model**

import ssl

ssl._create_default_https_context=ssl._create_unverified_context

# define model
model = fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2

# We need number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# check model on one image
model.eval()
output = model(batch_x[5].view(1, 3, 224, 224))


#GPU if avilable
model = model.to(device)

# define optimization 
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# set model to train
model.train()

# train model
for epoch in range(2):

    # initialize variables
    epoch_classif_loss = epoch_regress_loss = cnt = 0

    # loop through the data
    for batch_x, batch_y in data_loader:
        # get batch images and targets and transfer them to GPU if available
        batch_x = list(image.to(device) for image in batch_x)
        batch_y = [{k: v.to(device) for k, v in t.items()} for t in batch_y]

        # clear gradients
        optimizer.zero_grad()

        # pass images to model and get loss
        loss_dict = model(batch_x, batch_y)
        losses = sum(loss for loss in loss_dict.values())

        # do a backward pass
        losses.backward()

        # update gradients
        optimizer.step()

        # sum loss and get count
        epoch_classif_loss += loss_dict['loss_classifier'].item()
        epoch_regress_loss += loss_dict['loss_box_reg'].item()
        cnt += 1

    # take average loss for all batches
    epoch_classif_loss /= cnt
    epoch_regress_loss /= cnt
    
    # print loss
    print("Training loss for epoch {} is {} for classification and {} for regression "
        .format(epoch + 1, epoch_classif_loss, epoch_regress_loss)
    )

# **Model Evaluation**

tdata = PotHoleDataset('test/images/', transforms=transform,train=False)

# defining data loader
data_loader_test = torch.utils.data.DataLoader(
    tdata, 
    batch_size=8, 
    shuffle=False
)

my_submission = pd.DataFrame()
model.eval()
for batch_test,names in data_loader_test:
output = model(batch_test.to(device))
output = [{k: v.to("cpu") for k, v in t.items()} for t in output]
temp = pd.DataFrame()
for i in range(len(output)):
boxes = output[i]['boxes'].detach().numpy()
scores = output[i]['scores'].detach().numpy()
labels = output[i]['labels'].detach().numpy()
scores = np.expand_dims(scores, axis=1)
labels = np.expand_dims(labels, axis=1)
batch_df = pd.DataFrame(np.hstack((boxes,scores,labels)),columns=['XMin','YMin','XMax','YMax','Conf','LabelName'])
batch_df['ImageID'] = names[i]
img = cv2.imread('/content/test/images/'+names[i])
h,w = img.shape[:2]
batch_df['LabelName'] = 'pothole'
batch_df['XMax'] = batch_df['XMax']/224 * w
batch_df['XMin'] = batch_df['XMin']/224 * w
batch_df['YMax'] = batch_df['YMax']/224 * h
batch_df['YMin'] = batch_df['YMin']/224 * h
batch_df['XMax'] = batch_df['XMax'].astype(int)
batch_df['XMin'] = batch_df['XMin'].astype(int)
batch_df['YMax'] = batch_df['YMax'].astype(int)
batch_df['YMin'] = batch_df['YMin'].astype(int)
temp = pd.concat([batch_df,temp])
my_submission = pd.concat([my_submission,temp])

my_submission.to_csv('2_epoch.csv',index=False)