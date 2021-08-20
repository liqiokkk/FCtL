#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from utils.metrics import ConfusionMatrix
from PIL import Image, ImageOps
from models.fcn import FCN8, MiniFCN8


# torch.cuda.synchronize()
# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

transformer = transforms.Compose([
    transforms.ToTensor(),
])

def resize(images, shape, label=False):
    '''
    resize PIL images
    shape: (w, h)
    '''
    resized = list(images)
    for i in range(len(images)):
        if label:
            resized[i] = images[i].resize(shape, Image.NEAREST)
        else:
            resized[i] = images[i].resize(shape, Image.BILINEAR)
    return resized

def _mask_transform(mask):
    target = np.array(mask).astype('int32')
    return target

def masks_transform(masks, numpy=False):
    '''
    masks: list of PIL images
    '''
    targets = []
    for m in masks:
        targets.append(_mask_transform(m))
    targets = np.array(targets) 
    if numpy:
        return targets
    else:
        return torch.from_numpy(targets).long().cuda()

def images_transform(images):
    '''
    images: list of PIL images
    '''
    inputs = []
    for img in images:
        inputs.append(transformer(img))
    inputs = torch.stack(inputs, dim=0).cuda()
    return inputs

def get_patch_info(shape, p_size):
    '''
    shape: origin image size, (x, y)
    p_size: patch size (square)
    return: n_x, n_y, step_x, step_y
    '''
    x = shape[0]
    y = shape[1]
    n = m = 1
    while x > n * p_size:
        n += 1
    while p_size - 1.0 * (x - p_size) / (n - 1) < 50:
        n += 1
    while y > m * p_size:
        m += 1
    while p_size - 1.0 * (y - p_size) / (m - 1) < 50:
        m += 1
    return n, m, (x - p_size) * 1.0 / (n - 1), (y - p_size) * 1.0 / (m - 1)

def global2patch(images, p_size):
    '''
    image/label => patches
    p_size: patch size
    return: list of PIL patch images; coordinates: images->patches; ratios: (h, w)
    '''
    patches = []; coordinates = []; templates = []; sizes = []; ratios = [(0, 0)] * len(images); patch_ones = np.ones(p_size)
    for i in range(len(images)):
        w, h = images[i].size
        size = (h, w) 
        sizes.append(size)
        ratios[i] = (float(p_size[0]) / size[0], float(p_size[1]) / size[1])
        template = np.zeros(size)
        n_x, n_y, step_x, step_y = get_patch_info(size, p_size[0]) 
        patches.append([images[i]] * (n_x * n_y)) 
        coordinates.append([(0, 0)] * (n_x * n_y))
        for x in range(n_x):
            if x < n_x - 1: top = int(np.round(x * step_x))
            else: top = size[0] - p_size[0]
            for y in range(n_y):
                if y < n_y - 1: left = int(np.round(y * step_y))
                else: left = size[1] - p_size[1]
                template[top:top+p_size[0], left:left+p_size[1]] += patch_ones
                coordinates[i][x * n_y + y] = (1.0 * top / size[0], 1.0 * left / size[1])
                patches[i][x * n_y + y] = transforms.functional.crop(images[i], top, left, p_size[0], p_size[1]) #508 508
        templates.append(Variable(torch.Tensor(template).expand(1, 1, -1, -1)).cuda())
    return patches, coordinates, templates, sizes, ratios

def global2bigpatch(images, p_size, mul=2):
    if mul == 1.5:
        sz = int(p_size[0]/4)
    elif mul == 2:
        sz = int(p_size[0]/2)
    elif mul == 2.5:
        sz = int(p_size[0]*3/4)
    elif mul == 3:
        sz = int(p_size[0])
    elif mul == 4:    
        sz = int(p_size[0]*3/2)
    patches = []; coordinates = []; templates = []; sizes = []; ratios = [(0, 0)] * len(images); patch_ones = np.ones(p_size)
    for i in range(len(images)):
        w, h = images[i].size
        big = ImageOps.expand(images[i],(sz, sz, sz, sz),fill='black')
        size = (h, w) 
        n_x, n_y, step_x, step_y = get_patch_info(size, p_size[0]) 
        patches.append([big] * (n_x * n_y))
        for x in range(n_x):
            if x < n_x - 1: top = int(np.round(x * step_x))
            else: top = size[0] - p_size[0]
            for y in range(n_y):
                if y < n_y - 1: left = int(np.round(y * step_y))
                else: left = size[1] - p_size[1]
                patches[i][x * n_y + y] = transforms.functional.crop(big, top, left, int(p_size[0]*mul), int(p_size[1]*mul)).resize(p_size, Image.BILINEAR) #508 508
    return patches#, coordinates, templates, sizes, ratios

def patch2global(patches, n_class, sizes, coordinates, p_size):
    '''
    predicted patches (after classify layer) => predictions
    return: list of np.array
    '''
    predictions = [ np.zeros((n_class, size[0], size[1])) for size in sizes ]
    for i in range(len(sizes)):
        for j in range(len(coordinates[i])):
            top, left = coordinates[i][j]
            top = int(np.round(top * sizes[i][0])); left = int(np.round(left * sizes[i][1]))
            predictions[i][:, top: top + p_size[0], left: left + p_size[1]] += patches[i][j]
    return predictions

def collate(batch):
    image = [ b['image'] for b in batch ] # w, h
    label = [ b['label'] for b in batch ]
    id = [ b['id'] for b in batch ]
    return {'image': image, 'label': label, 'id': id}

def collate_test(batch):
    image = [ b['image'] for b in batch ] # w, h
    id = [ b['id'] for b in batch ]
    return {'image': image, 'id': id}


def create_model_load_weights(n_class, pre_path="", glo_path_10="", glo_path_15="", mode=1):
    model = FCN8(n_class, mode)
    model = nn.DataParallel(model)
    model = model.cuda()

    if pre_path != './saved_models_1/':
        print('prepareing model...') 
        # load fixed basic global branch
        partial = torch.load(pre_path)
        state = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in partial.items() if k in state}
        # 2. overwrite entries in the existing state dict
        state.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(state)
        
    global_fixed_10 = None
    if mode == 2 or mode == 3:
        print('prepareing global_10 model...') 
        global_fixed_10 = MiniFCN8(n_class)
        global_fixed_10 = nn.DataParallel(global_fixed_10)
        global_fixed_10 = global_fixed_10.cuda()
        if glo_path_10 != './saved_models_1/':
            partial = torch.load(glo_path_10)
            state = global_fixed_10.state_dict()
            pretrained_dict = {k: v for k, v in partial.items() if k in state}
            state.update(pretrained_dict)
            global_fixed_10.load_state_dict(state)
        global_fixed_10.eval()
        
    global_fixed_15 = None
    if mode == 3:
        print('prepareing global_15 model...') 
        global_fixed_15 = MiniFCN8(n_class)
        global_fixed_15 = nn.DataParallel(global_fixed_15)
        global_fixed_15 = global_fixed_15.cuda()
        if glo_path_15 != './saved_models_1/':
            partial = torch.load(glo_path_15)
            state = global_fixed_15.state_dict()
            pretrained_dict = {k: v for k, v in partial.items() if k in state}
            state.update(pretrained_dict)
            global_fixed_15.load_state_dict(state)
        global_fixed_15.eval()
    return model, global_fixed_10, global_fixed_15


def get_optimizer(model, learning_rate=2e-5):
    optimizer = torch.optim.Adam([
    {'params': model.module.parameters(), 'lr': learning_rate},
    ], weight_decay=5e-4)
    return optimizer

class Trainer(object):
    def __init__(self, criterion, optimizer, n_class, size_p, size_g, sub_batch_size=6, mode=1, dataset=1, context10=2, context15=3):
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = ConfusionMatrix(n_class)
        self.n_class = n_class
        self.size_p = size_p
        self.size_g = size_g
        self.sub_batch_size = sub_batch_size
        self.mode = mode
        self.context10 = context10
        self.context15 = context15
        
    def set_train(self, model):
        model.module.train()
    def get_scores(self):
        score = self.metrics.get_scores()
        return score

    def reset_metrics(self):
        self.metrics.reset()

    def train(self, sample, model, global_fixed_10, global_fixed_15):
        images, labels = sample['image'], sample['label'] # PIL images
        labels_npy = masks_transform(labels, numpy=True) # label of origin size in numpy 
        
        patches, coordinates, templates, sizes, ratios = global2patch(images, self.size_p)
        label_patches, _, _, _, _ = global2patch(labels, self.size_p)
        predicted_patches = [ np.zeros((len(coordinates[i]), self.n_class, self.size_p[0], self.size_p[1])) for i in range(len(images)) ] 
        
        ##################1 2 3
        if self.mode != 1:
            big_patches_10 = global2bigpatch(images, self.size_p, self.context10)
        if self.mode == 3:
                big_patches_15 = global2bigpatch(images, self.size_p, self.context15)
        pool5_10, pool5_15 = None, None
        # training with patches ###########################################
        for i in range(len(images)):
            j = 0
            while j < len(coordinates[i]):
                patches_var = images_transform(patches[i][j : j+self.sub_batch_size]) # b, c, h, w
                label_patches_var = masks_transform(label_patches[i][j : j+self.sub_batch_size])     
                big_patches_10_var=None
                if self.mode != 1:
                    big_patches_10_var = images_transform(big_patches_10[i][j : j+self.sub_batch_size])
                if self.mode == 3:
                    with torch.no_grad():
                        pool5_10 = global_fixed_10.forward(big_patches_10_var)
                        big_patches_15_var = images_transform(big_patches_15[i][j : j+self.sub_batch_size])
                        pool5_15 = global_fixed_15.forward(big_patches_15_var)
                if self.mode == 1 or self.mode == 2:
                    output_patches = model.forward(patches_var, y=big_patches_10_var)  
                else:
                     output_patches = model.forward(patches_var, pool5_10, pool5_15)
                loss = self.criterion(output_patches, label_patches_var)
                loss.backward()
                # patch predictions
                predicted_patches[i][j:j+output_patches.size()[0]] = F.interpolate(output_patches, size=self.size_p, mode='nearest').data.cpu().numpy()
                j += self.sub_batch_size
        self.optimizer.step()
        self.optimizer.zero_grad()
        ####################################################################################
        scores = np.array(patch2global(predicted_patches, self.n_class, sizes, coordinates, self.size_p)) # merge softmax scores from patches (overlaps)
        predictions = scores.argmax(1) # b, h, w
        self.metrics.update(labels_npy, predictions)
        return loss


class Evaluator(object):
    def __init__(self, n_class, size_p, size_g, sub_batch_size=6, mode=1, val=True, dataset=1, context10=2, context15=3):
        self.metrics = ConfusionMatrix(n_class)
        self.n_class = n_class
        self.size_p = size_p
        self.size_g = size_g
        self.sub_batch_size = sub_batch_size
        self.mode = mode
        self.val = val
        self.context10 = context10
        self.context15 = context15
        if not val:
            self.flip_range = [False, True]
            self.rotate_range = [0, 1, 2, 3]
        else:
            self.flip_range = [False]
            self.rotate_range = [0]
    
    def get_scores(self):
        score = self.metrics.get_scores()
        return score

    def reset_metrics(self):
        self.metrics.reset()

    def eval_test(self, sample, model, global_fixed_10, global_fixed_15):
        with torch.no_grad():
            images = sample['image']
            if self.val:
                labels = sample['label'] # PIL images
                labels_npy = masks_transform(labels, numpy=True)
                
            images = [ image.copy() for image in images ]
            scores = [ np.zeros((1, self.n_class, images[i].size[1], images[i].size[0])) for i in range(len(images)) ]
            for flip in self.flip_range:
                if flip:
                    # we already rotated images for 270'
                    for b in range(len(images)):
                        images[b] = transforms.functional.rotate(images[b], 90) # rotate back!
                        images[b] = transforms.functional.hflip(images[b])
                for angle in self.rotate_range:
                    if angle > 0:
                        for b in range(len(images)):
                            images[b] = transforms.functional.rotate(images[b], 90)
                    # prepare global images onto cuda

                    patches, coordinates, templates, sizes, ratios = global2patch(images, self.size_p)
                    predicted_patches = [ np.zeros((len(coordinates[i]), self.n_class, self.size_p[0], self.size_p[1])) for i in range(len(images)) ]
                    
                    if self.mode == 2 or self.mode == 3:
                        big_patches_10 = global2bigpatch(images, self.size_p, self.context10)
                    if self.mode == 3:
                        big_patches_15 = global2bigpatch(images, self.size_p, self.context15)
                    # eval with patches ###########################################
                    for i in range(len(images)):
                        j = 0
                        while j < len(coordinates[i]):
                            patches_var = images_transform(patches[i][j : j+self.sub_batch_size]) # b, c, h, w
                            big_patches_10_var = None
                            if self.mode == 2 or self.mode == 3:
                                big_patches_10_var = images_transform(big_patches_10[i][j : j+self.sub_batch_size])
                                
                            if self.mode == 1 or self.mode == 2:
                                output_patches = model.forward(patches_var, y=big_patches_10_var) 
                            else: ##3
                                pool5_10 = global_fixed_10.forward(big_patches_10_var)
                                big_patches_15_var = images_transform(big_patches_15[i][j : j+self.sub_batch_size])
                                pool5_15 = global_fixed_15.forward(big_patches_15_var)
                                output_patches = model.forward(patches_var, pool5_10, pool5_15)
                            # patch predictions
                            predicted_patches[i][j:j+output_patches.size()[0]] += F.interpolate(output_patches, size=self.size_p, mode='nearest').data.cpu().numpy()
                            j += patches_var.size()[0]
                        if flip:
                            scores[i] += np.flip(np.rot90(np.array(patch2global(predicted_patches[i:i+1], self.n_class, sizes[i:i+1], coordinates[i:i+1], self.size_p)), k=angle, axes=(3, 2)), axis=3) # merge softmax scores from patches (overlaps)
                        else:
                            scores[i] += np.rot90(np.array(patch2global(predicted_patches[i:i+1], self.n_class, sizes[i:i+1], coordinates[i:i+1], self.size_p)), k=angle, axes=(3, 2)) # merge softmax scores from patches (overlaps)
                    ###############################################################

            # patch predictions ###########################
            predictions = [ score.argmax(1)[0] for score in scores ]
            if self.val:
                self.metrics.update(labels_npy, predictions)
            ###################################################
            return predictions

