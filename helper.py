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

def global2bigpatch(images, p_size, context=3):
    if context == 3:
        sz = p_size[0]
    elif context == 1:
        sz = 0
    elif context == 2:
        sz = int(p_size[0]/2)
    elif context == 4:
        sz = int(p_size[0]/4)
        
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
                patches[i][x * n_y + y] = transforms.functional.crop(big, top, left, int(p_size[0]*context), int(p_size[1]*context)).resize(p_size, Image.BILINEAR) #508 508
    return patches

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


def create_model_load_weights(n_class, pre_path=None, glo_path=None, c_path=None, mode=1):

    model = FCN8(n_class, mode)
    model = nn.DataParallel(model)
    model = model.cuda()
    if pre_path != './saved_models/':
        print('prepareing pre model...')
        # load fixed basic global branch
        partial = torch.load(pre_path)
        state = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in partial.items() if k in state}
        # 2. overwrite entries in the existing state dict
        state.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(state)
        
    c_fixed = None
    if mode == 4:
        print('prepareing context model...')
        c_fixed = MiniFCN8(n_class)
        c_fixed = nn.DataParallel(c_fixed)
        c_fixed = c_fixed.cuda()
        if c_path != './saved_models/':
            partial = torch.load(c_path)
            state = c_fixed.state_dict()
            pretrained_dict = {k: v for k, v in partial.items() if k in state}
            state.update(pretrained_dict)
            c_fixed.load_state_dict(state)
        c_fixed.eval()
        
    global_fixed = None
    if mode == 3 or mode == 4:
        print('prepareing global model...')
        global_fixed = FCN8(n_class, 0)
        global_fixed = nn.DataParallel(global_fixed)
        global_fixed = global_fixed.cuda()
        if glo_path != './saved_models/':
            partial = torch.load(glo_path)
            state = global_fixed.state_dict()
            pretrained_dict = {k: v for k, v in partial.items() if k in state}
            state.update(pretrained_dict)
            global_fixed.load_state_dict(state)
        global_fixed.eval()
    
    return model, c_fixed, global_fixed


def get_optimizer(model, learning_rate=2e-5):
    optimizer = torch.optim.Adam([
    {'params': model.module.parameters(), 'lr': learning_rate},
    ], weight_decay=5e-4)
    return optimizer

class Trainer(object):
    def __init__(self, criterion, optimizer, n_class, size_p, size_g, sub_batch_size=6, mode=1, dataset=1, context=3):
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = ConfusionMatrix(n_class)
        self.n_class = n_class
        self.size_p = size_p
        self.size_g = size_g
        self.sub_batch_size = sub_batch_size
        self.mode = mode
        self.context = context
        
    def set_train(self, model):
        model.module.train()
    def get_scores(self):
        score = self.metrics.get_scores()
        return score

    def reset_metrics(self):
        self.metrics.reset()

    def train(self, sample, model, c_fixed, global_fixed):
        images, labels = sample['image'], sample['label'] # PIL images
        labels_npy = masks_transform(labels, numpy=True) # label of origin size in numpy
        
        patches, coordinates, templates, sizes, ratios = global2patch(images, self.size_p)
        label_patches, _, _, _, _ = global2patch(labels, self.size_p)
        predicted_patches = [ np.zeros((len(coordinates[i]), self.n_class, self.size_p[0], self.size_p[1])) for i in range(len(images)) ] 
        
        images_glb = resize(images, self.size_g) # list of resized PIL images
        images_glb = images_transform(images_glb)        
        if self.mode == 0:
            labels_glb = resize(labels, self.size_g, label=True)
            labels_glb = masks_transform(labels_glb)
            outputs_global = model.forward(images_glb)
            loss = self.criterion(outputs_global, labels_glb)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        else: #1 2 3 4
            pool5_c, pool5 = None, None
            if self.mode == 2 or self.mode == 4:
                big_patches = global2bigpatch(images, self.size_p, self.context)
            for i in range(len(images)):
                if self.mode == 3 or self.mode == 4:
                    with torch.no_grad():
                        _, _, pool5 = global_fixed.forward(images_glb[i:i+1], 1)
                j = 0
                while j < len(coordinates[i]):
                    patches_var = images_transform(patches[i][j : j+self.sub_batch_size]) # b, c, h, w
                    label_patches_var = masks_transform(label_patches[i][j : j+self.sub_batch_size])
                    # big
                    big_patches_var = None
                    if self.mode == 2 or self.mode == 4:
                        big_patches_var = images_transform(big_patches[i][j : j+self.sub_batch_size]) # b, c, h, w
                        if self.mode == 4:
                            with torch.no_grad():
                                pool5_c = c_fixed.forward(big_patches_var)
                    if self.mode == 1 or self.mode == 2:
                        output_patches = model.forward(patches_var, y=big_patches_var)      
                    else:
                        output_patches = model.forward(patches_var, 0, pool5_c, pool5)
                    loss = self.criterion(output_patches, label_patches_var)
                    loss.backward()#retain_graph=True)
                    # patch predictions
                    predicted_patches[i][j:j+output_patches.size()[0]] = F.interpolate(output_patches, size=self.size_p, mode='nearest').data.cpu().numpy()
                    j += self.sub_batch_size
            self.optimizer.step()
            self.optimizer.zero_grad()
        # patch predictions ###########################
        if self.mode == 0:
            pass
        else:
            scores = np.array(patch2global(predicted_patches, self.n_class, sizes, coordinates, self.size_p)) # merge softmax scores from patches (overlaps)
            predictions = scores.argmax(1) # b, h, w
        self.metrics.update(labels_npy, predictions)
        return loss


class Evaluator(object):
    def __init__(self, n_class, size_p, size_g, sub_batch_size=6, mode=1, val=True, dataset=1 , context=3):
        self.metrics = ConfusionMatrix(n_class)
        self.n_class = n_class
        self.size_p = size_p
        self.size_g = size_g
        self.sub_batch_size = sub_batch_size
        self.mode = mode
        self.val = val
        self.context = context

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

    def eval_test(self, sample, model, c_fixed, global_fixed):
        with torch.no_grad():
            images = sample['image']
            if self.val:
                labels = sample['label'] # PIL images
                labels_npy = masks_transform(labels, numpy=True)
            images_global = resize(images, self.size_g)
            if self.mode == 0:
                pass
            # 1 2 3 4                                
            images = [ image.copy() for image in images ]
            scores = [ np.zeros((1, self.n_class, images[i].size[1], images[i].size[0])) for i in range(len(images)) ]
            for flip in self.flip_range:
                if flip:
                    # we already rotated images for 270'
                    for b in range(len(images)):
                        images[b] = transforms.functional.rotate(images[b], 90) # rotate back!
                        images[b] = transforms.functional.hflip(images[b])
                        if self.mode == 3 or self.mode == 4:
                            images_global[b] = transforms.functional.rotate(images_global[b], 90) # rotate back!
                            images_global[b] = transforms.functional.hflip(images_global[b])
                for angle in self.rotate_range:
                    if angle > 0:
                        for b in range(len(images)):
                            images[b] = transforms.functional.rotate(images[b], 90)
                            if self.mode == 3 or self.mode == 4:
                                images_global[b] = transforms.functional.rotate(images_global[b], 90)
                    # prepare global images onto cuda

                    patches, coordinates, templates, sizes, ratios = global2patch(images, self.size_p)
                    predicted_patches = [ np.zeros((len(coordinates[i]), self.n_class, self.size_p[0], self.size_p[1])) for i in range(len(images)) ]
                    if self.mode == 2 or self.mode == 4:
                        big_patches = global2bigpatch(images, self.size_p, self.context)
                    if self.mode == 3 or self.mode == 4:
                        images_glb = images_transform(images_global)
                    # eval with patches ###########################################
                    for i in range(len(images)):
                        pool5 = None
                        pool5_c = None
                        if self.mode == 3 or self.mode == 4:
                            _, _, pool5 = global_fixed.forward(images_glb[i:i+1], 1)
                        j = 0
                        while j < len(coordinates[i]):
                            patches_var = images_transform(patches[i][j : j+self.sub_batch_size]) # b, c, h, w
                            big_patches_var = None
                            if self.mode == 2 or self.mode == 4:
                                big_patches_var = images_transform(big_patches[i][j : j+self.sub_batch_size])
                                if self.mode == 4:
                                    pool5_c = c_fixed.forward(big_patches_var)
                            if self.mode == 1 or self.mode == 2:
                                pass
                            else:  
                                output_patches = model.forward(patches_var, 0, pool5_c, pool5)
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


