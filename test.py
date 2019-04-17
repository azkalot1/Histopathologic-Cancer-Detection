import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils import data
import pretrainedmodels
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import os
import cv2
from skimage.io import imread
from torch.utils.data.sampler import WeightedRandomSampler, BatchSampler
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, Normalize, RandomGamma, RandomBrightnessContrast, HueSaturationValue, CLAHE, ChannelShuffle,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, PadIfNeeded, RandomCrop, Resize
)
from tqdm import tqdm_notebook
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold
import matplotlib.pyplot as plt
import pretrainedmodels.utils as utils
from sklearn.metrics import roc_auc_score
use_cuda = torch.cuda.is_available()
import pickle

'''
Get predictions for each fold (OOF preds) and test data using TTA
'''

device = torch.device("cuda" if use_cuda else "cpu")
DIR = './'
train_im_dir = DIR+'/train'
test_im_dir = DIR+'/test'
model_dir = os.path.join(DIR,'resnet34')
model_name = 'resnet34'
train_data = pd.read_csv(os.path.join(DIR,'train_labels.csv'))
test_data = pd.read_csv(os.path.join(DIR,'sample_submission.csv'))
patch_ids = pd.read_csv(os.path.join(DIR,'patch_id_wsi.csv'))
train_data = pd.merge(train_data, patch_ids, on='id')
n_groups = 15
skf = GroupKFold(n_splits=n_groups)
folds_id_train = []
folds_label_train = []
folds_id_val = []
folds_label_val = []
for train_index, test_index in skf.split(train_data['id'].values, train_data['label'].values, train_data['wsi'].values):
    folds_id_train.append(train_data['id'].values[train_index])
    folds_id_val.append(train_data['id'].values[test_index])
    folds_label_train.append(train_data['label'].values[train_index])
    folds_label_val.append(train_data['label'].values[test_index])
test_id = test_data['id'].values
test_label = test_data['label'].values



val_preds = []
val_labels = []
test_preds = []
scores_CV = []
for valid_idx in range(n_groups):
    base_model = pretrainedmodels.resnet34(num_classes=1000,pretrained='imagenet').to(device) #load pretrained as base
    model = Net(base_model, 512).to(device) # create model
    model.load_state_dict(torch.load(model_dir+'/resnet34.fold{}.best.pt'.format(valid_idx))) #loading weights
    model.eval()
    valid_preds_idx = np.zeros((len(folds_id_val[valid_idx])))
    valid_target_idx = np.zeros((len(folds_id_val[valid_idx])))
    test_preds_idx = np.zeros((len(test_label)))
    val_loader = torch.utils.data.DataLoader(DataGenerator(folds_id_val[valid_idx], folds_label_val[valid_idx], validation_aug, train_im_dir), 
                                             shuffle=False, pin_memory=False, num_workers=1,batch_size=1)
    test_loader = torch.utils.data.DataLoader(DataGenerator(test_id, test_label, validation_aug, test_im_dir), 
                                              shuffle=False, pin_memory=False, num_workers=1,batch_size=1)  
    #predction for validation data
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(tqdm_notebook(val_loader)):
            #output = protein_model(x.to(device, dtype=torch.float))
            image = np.rollaxis(x.numpy()[0], 0, 3)
            images = make_tta_heavy(image,n_images=8) #create 8 images for random augmentations to take mean prediciton of each
            output = model(torch.from_numpy(images).to(device, dtype=torch.float))
            output = output.mean()
            valid_preds_idx[batch_idx] = output
            valid_target_idx[batch_idx] = target
    val_preds.append(valid_preds_idx)
    val_labels.append(valid_target_idx)
    score_CV_idx = roc_auc_score(valid_target_idx, valid_preds_idx)
    scores_CV.append(score_CV_idx)
    print('fold {}, score {}'.format(valid_idx, score_CV_idx))
    #predction for test data
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(tqdm_notebook(test_loader)):
            image = np.rollaxis(x.numpy()[0], 0, 3)
            images = make_tta_heavy(image,n_images=8)
            output = model(torch.from_numpy(images).to(device, dtype=torch.float))
            output = output.mean()            
            #output = protein_model(x.to(device, dtype=torch.float))
            test_preds_idx[batch_idx] = output
    test_preds.append(test_preds_idx)    
    
    
    
val_preds_combined = np.hstack(val_preds)
val_labels_combined = np.hstack(val_labels)
#average test predctions over each fold
test_preds_combined = np.vstack(test_preds)
test_preds_combined = np.mean(test_preds_combined, axis=0)
cv_rocauc = roc_auc_score(val_labels_combined, val_preds_combined
print('Total roc auc'.format(cv_rocauc))
d = dict({'oof_preds':val_preds_combined,'oof_labels':val_labels_combined,'test_preds':test_preds_combined})
with open(os.path.join(model_dir, "resnet34.tta.Preds.pickle"), "wb") as output_file:
    pickle.dump(d, output_file)
#create sample submission
test_data['label'] = d['test_preds']
test_data.to_csv(os.path.join(model_dir, 'resnet34.prediction.tta.csv') ,sep=',',index=False)
