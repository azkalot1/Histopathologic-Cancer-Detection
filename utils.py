import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.optim import Optimizer
from torch.utils import data
import pretrainedmodels
import numpy as np
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
import pretrainedmodels.utils as utils
from sklearn.metrics import roc_auc_score


def train(model, train_loader, optimizer, epoch, log_interval, loss_f, samples_per_epoch, device, cycling_optimizer=False):
    """Trains the model using the provided optimizer and loss function.
    Shows output each log_interval iterations 
    Args:
        model: Pytorch model to train.
        train_loader: Data loader.
        optimizer: pytroch optimizer.
        epoch: Current epoch.
        log_interval: Show model training progress each log_interval steps.
        loss_f: Loss function to optimize.
        samples_per_epoch: Number of samples per epoch to scale loss.
        device: pytorch device
        cycling_optimizer: Indicates of optimizer is cycling.
    """
    model.train()
    total_losses = []
    losses =[]
    for batch_idx, (x, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(x.to(device, dtype=torch.float))
        loss = loss_f(output, target.to(device, dtype=torch.float))
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), 4)
        if cycling_optimizer:
            optimizer.batch_step()
        else:
            optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.3f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), samples_per_epoch,
                100. * batch_idx * len(x) / samples_per_epoch, np.mean(losses)))
            total_losses.append(np.mean(losses))
            losses = []
    train_loss_mean = np.mean(total_losses)
    print('Mean train loss on epoch {} : {}'.format(epoch, train_loss_mean))
    return train_loss_mean
            
def test(model, test_loader, loss_f, epoch, device):
    """Test the model with validation data.
    Args:
        model: Pytorch model to test data with.
        test_loader: Data loader.
        loss_f: Loss function.
        epoch: Current epoch.
        device: pytorch device        
    """
    model.eval()
    test_loss = 0
    predictions=[]
    targets=[]
    test_loss=[]
    with torch.no_grad():
        for x, target in test_loader:
            output = model(x.to(device, dtype=torch.float))
            test_loss.append(loss_f(output, target.to(device, dtype=torch.float)).item())
            predictions.append(output.cpu())
            targets.append(target.cpu())
    predictions = np.vstack(predictions)
    targets = np.vstack(targets)
    score = roc_auc_score(targets, predictions)
    test_loss  = np.mean(test_loss)
    print('\nTest set: Average loss: {:.6f}, roc auc: {:.4f}\n'.format(test_loss, score))
    return test_loss, score	

class Net(nn.Module):
    """Build the nn network based on pretrained resnet models.
    Args:
        base_model: resnet34\resnet50\etc from pretrained models
        n_features: n features from last pooling layer       
    """
    def __init__(self, base_model, n_features):
        super(Net, self).__init__()
        self.layer0 = nn.Sequential(*list(base_model.children())[:4])
        self.layer1 = nn.Sequential(*list(base_model.layer1))
        self.layer2 = nn.Sequential(*list(base_model.layer2))
        self.layer3 = nn.Sequential(*list(base_model.layer3))
        self.layer4 = nn.Sequential(*list(base_model.layer4))
        self.dense1 = nn.Sequential(nn.Linear(n_features, 128))
        self.dense2 = nn.Sequential(nn.Linear(128, 64))
        self.classif = nn.Sequential(nn.Linear(64, 1))
    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.classif(x)
        x = torch.sigmoid(x)
        return x
    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x 
    
    

class CyclicLR(object):
    """Sets the learning rate of each parameter group according to
    cyclical learning rate policy (CLR). The policy cycles the learning
    rate between two boundaries with a constant frequency, as detailed in
    the paper `Cyclical Learning Rates for Training Neural Networks`_.
    The distance between the two boundaries can be scaled on a per-iteration
    or per-cycle basis.
    Cyclical learning rate policy changes the learning rate after every batch.
    `batch_step` should be called after a batch has been used for training.
    To resume training, save `last_batch_iteration` and use it to instantiate `CycleLR`.
    This class has three built-in policies, as put forth in the paper:
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    This implementation was adapted from the github repo: `bckenstler/CLR`_
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        base_lr (float or list): Initial learning rate which is the
            lower boundary in the cycle for eachparam groups.
            Default: 0.001
        max_lr (float or list): Upper boundaries in the cycle for
            each parameter group. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function. Default: 0.006
        step_size (int): Number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch. Default: 2000
        mode (str): One of {triangular, triangular2, exp_range}.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
            Default: 'triangular'
        gamma (float): Constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
            Default: 1.0
        scale_fn (function): Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
            Default: None
        scale_mode (str): {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle).
            Default: 'cycle'
        last_batch_iteration (int): The index of the last batch. Default: -1
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.CyclicLR(optimizer)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         scheduler.batch_step()
        >>>         train_batch(...)
    .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    .. _bckenstler/CLR: https://github.com/bckenstler/CLR
    """

    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration

    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs

def write_log(logfile, train_loss, test_loss, test_score, lr):
    with open(logfile, "a+") as log:
        log.write("{}\t{}\t{}\t{}\n".format(train_loss, test_loss, test_score, lr))
        
        
        
def aug_train(p=1): 
    return Compose([Resize(224, 224), 
                    HorizontalFlip(), 
                    VerticalFlip(), 
                    RandomRotate90(), 
                    Transpose(), 
                    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
                    OpticalDistortion(),
                    GridDistortion(), 
                    RandomBrightnessContrast(p=0.3), 
                    RandomGamma(p=0.3), 
                    OneOf([HueSaturationValue(hue_shift_limit=20, sat_shift_limit=0.1, val_shift_limit=0.1, p=0.3), 
                           ChannelShuffle(p=0.3), CLAHE(p=0.3)])], p=p)
def aug_val(p=1):
    return Compose([
        Resize(224, 224)
    ], p=p)


class DataGenerator(data.Dataset):
    """Generates dataset for loading.
    Args:
        ids: images ids
        labels: labels of images (1/0)
        augment: image augmentation from albumentations
        imdir: path tpo folder with images
    """
    def __init__(self, ids, labels, augment, imdir):
        'Initialization'
        self.ids, self.labels = ids, labels
        self.augment = augment
        self.imdir = imdir
        
    def __len__(self):
        return len(self.ids) 

    def __getitem__(self, idx):
        imid = self.ids[idx]
        y = self.labels[idx]
        X = self.__load_image(imid)
        return X, np.expand_dims(y,0)

    def __load_image(self, imid):
        imid = imid+'.tif'
        im = imread(os.path.join(self.imdir, imid))
        if self.augment!=None:
            augmented = self.augment(image=im)
            im = augmented['image']
        im = im/255.0
        im = np.rollaxis(im, -1)
        return im     
    
    
def make_tta(image):
    '''
    return 4 pictures  - original, 3*90 rotations, mirror
    '''
    image_tta = np.zeros((4, image.shape[0], image.shape[1], 3))
    image_tta[0] = image
    aug = HorizontalFlip(p=1)
    image_aug = aug(image=image)['image']
    image_tta[1] = image_aug
    aug = VerticalFlip(p=1)
    image_aug = aug(image=image)['image']
    image_tta[2] = image_aug
    aug = Transpose(p=1)
    image_aug = aug(image=image)['image']
    image_tta[3] = image_aug    
    image_tta = np.rollaxis(image_tta, -1, 1)
    return image_tta
def aug_train_heavy(p=1):
    return Compose([HorizontalFlip(), VerticalFlip(), RandomRotate90(), Transpose(), RandomBrightnessContrast(p=0.3), RandomGamma(p=0.3), OneOf([HueSaturationValue(hue_shift_limit=20, sat_shift_limit=0.1, val_shift_limit=0.1, p=0.3), ChannelShuffle(p=0.3)])], p=p)
heavy_tta = aug_train_heavy()

def make_tta_heavy(image, n_images=12):
    image_tta = np.zeros((n_images, image.shape[0], image.shape[1], 3))
    image_tta[0] = image/255.0
    for i in range(1,n_images):
        image_aug = heavy_tta(image=image)['image']
        image_tta[i] = image_aug/255.0
    image_tta = np.rollaxis(image_tta, -1, 1)
    return image_tta 