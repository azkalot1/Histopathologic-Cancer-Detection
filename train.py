import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.optim import Optimizer
from torch.utils import data
import pretrainedmodels
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import os
import cv2
from skimage.io import imread
from torch.utils.data.sampler import WeightedRandomSampler, BatchSampler
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold
import pretrainedmodels.utils as utils
from sklearn.metrics import roc_auc_score
from utils import *




DIR = './' #folder with train and test data
train_im_dir = DIR+'/train'
test_im_dir = DIR+'/test'
train_data = pd.read_csv(os.path.join(DIR,'train_labels.csv')) #labels for train data
patch_ids = pd.read_csv(os.path.join(DIR,'patch_id_wsi.csv')) #slides id for correct split 
model_dir = os.path.join(DIR, 'resnet34')
model_name = 'resnet34'
n_groups = 15 # number of folds
b_size = 96 # batch size


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
train_data = pd.merge(train_data, patch_ids, on='id')
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
samples_per_epoch = 50000 #define number of samples per epoch, since dataset is big
for valid_idx in range(n_groups):
    logfile =  model_dir+'/{}.fold{}.logfile.txt'.format(model_name, valid_idx)
    best_w_path = model_dir+'/{}.fold{}.best.pt'.format(model_name, valid_idx)
    es_w_path =  model_dir+'/{}.fold{}.es.pt'.format(model_name, valid_idx)
    print('Training fold {}'.format(valid_idx))
    with open(logfile, "w") as log:
        pass    
    traing_aug = aug_train()
    validation_aug = aug_val()
    curr_lr = 3e-4
    train_sampler = torch.utils.data.RandomSampler(DataGenerator(folds_id_val[valid_idx], 
                                                                 folds_label_val[valid_idx], 
                                                                 validation_aug, train_im_dir),
                                                   replacement=True, 
                                                   num_samples=samples_per_epoch)
    train_loader = torch.utils.data.DataLoader(DataGenerator(folds_id_train[valid_idx], 
                                                             folds_label_train[valid_idx], 
                                                             traing_aug, train_im_dir),
                                               pin_memory=False,
                                               num_workers=4,
                                               batch_size=b_size, 
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(DataGenerator(folds_id_val[valid_idx], 
                                                       folds_label_val[valid_idx], 
                                                       validation_aug, train_im_dir),
                                             pin_memory=False,
                                             num_workers=1,
                                             batch_size=b_size)
    loss_f = nn.BCELoss()
    best_score = 0
    best_loss = 1e5
    idx_stop = 0
    #load pretrained resnet34 model
    base_model = pretrainedmodels.resnet34(num_classes=1000, 
                                           pretrained='imagenet').to(device)
    model = Net(base_model, 512).to(device)
    #training with frozen layers except for classfication head 
    optimizer = optim.SGD([{'params': model.layer0.parameters(), 'lr': 0},
                           {'params': model.layer1.parameters(), 'lr': 0},
                           {'params': model.layer2.parameters(), 'lr': 0},
                           {'params': model.layer3.parameters(), 'lr': 0},
                           {'params': model.layer4.parameters(), 'lr': 0},
                           {'params': model.classif.parameters()}], lr=0.05, momentum=0.9)
    train_loss = train(model, train_loader, optimizer, 0, 100, loss_f, samples_per_epoch)
    test_loss, score = test(model, val_loader, loss_f, 0)
    write_log(logfile, train_loss, test_loss, score)
    '''
    start training the model with all layers
    Training scheme : train while validation loss decreases, save model at each improvement of test loss. 
    if loss does not decreases for 3 epochs, reload last best model, reduce lr by factor of 2. 
    If loss still doesn't decrease for 10 epochs, stop the model. 
    '''
    for epoch in range(50):
        optimizer = torch.optim.SGD(model.parameters(), lr=curr_lr, momentum=0.9)
        scheduler = CyclicLR(optimizer, max_lr=3*curr_lr)
        train_loss = train(model, train_loader, optimizer, epoch, 100, loss_f, samples_per_epoch)
        test_loss, score = test(model, val_loader, loss_f, epoch)
        write_log(logfile, train_loss, test_loss, score)
        if test_loss<best_loss:
            print('Test loss improved from {} to {}, saving'.format(best_loss, test_loss))
            best_loss = test_loss
            torch.save(model.state_dict(), best_w_path)
            idx_stop = 0
        else:
            print('Loss {}, did not improve from {} for {} epochs'.format(test_loss, best_loss, idx_stop))
            idx_stop += 1
        if idx_stop>3:
            print('Reducing LR by two and reloading best model')
            model.load_state_dict(torch.load(best_w_path))
            curr_lr = curr_lr/2
        if idx_stop>10:
            print('Stopping the model')
            torch.save(model.state_dict(), es_w_path)
