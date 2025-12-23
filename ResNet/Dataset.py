import os
import lmdb
import torch

import Utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pickle
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from Utils import draw_three, off2abs
from Hyper_params import hp
from SketchUtils import SketchUtil
from PIL import Image

class Quickdraw414k(data.Dataset):

    def __init__(self, mode='Train'):
        self.mode = mode
        if mode == 'Train':
            sketch_list = "../QuickDraw414k/picture_files/tiny_train_set.txt"
            path_root1 = '../QuickDraw414k/picture_files/train'
            path_root2 = '../QuickDraw414k/coordinate_files/train'
        elif mode == 'Test':
            sketch_list = "../QuickDraw414k/picture_files/tiny_test_set.txt"
            path_root1 = '../QuickDraw414k/picture_files/test'
            path_root2 = '../QuickDraw414k/coordinate_files/test'
        elif mode == 'Valid':
            sketch_list = "../QuickDraw414k/picture_files/tiny_val_set.txt"
            path_root1 = '../QuickDraw414k/picture_files/val'
            path_root2 = '../QuickDraw414k/coordinate_files/val'

        with open(sketch_list) as sketch_url_file:
            sketch_url_list = sketch_url_file.readlines()
            self.img_urls = [os.path.join(path_root1, sketch_url.strip().split(' ')[
                0]) for sketch_url in sketch_url_list]
            self.coordinate_urls = [os.path.join(path_root2, (sketch_url.strip(
            ).split(' ')[0]).replace('png', 'npy')) for sketch_url in sketch_url_list]

            self.labels = [int(sketch_url.strip().split(' ')[-1])
                           for sketch_url in sketch_url_list]
        print('Total ' + mode + ' Sample {}'.format(len(self.labels)))

        self.train_transform = get_ransform('Train')
        self.valid_transform = get_ransform('Valid')
        self.test_transform = get_ransform('Test')

    def __len__(self):
        return len(self.img_urls)

    def __getitem__(self, item):
        sketch_url = self.img_urls[item]
        coordinate_url = self.coordinate_urls[item]
        label = self.labels[item]
        # img = Image.open(sketch_url, 'r').resize((224, 224))

        seq = np.load(coordinate_url, encoding='latin1', allow_pickle=True)
        if seq.dtype == 'object':
            seq = seq[0]
        assert seq.shape == (100, 4)
        seq = seq.astype('float32')
        seq = seq[:, 0:3]
        index_neg = np.where(seq == -1)[0]
        

        if len(index_neg) == 0:
            seq = off2abs(seq)

            if random.uniform(0, 1) > 0.5 and self.mode == 'Train':
               seq[:, 0:2] = SketchUtil.random_affine_transform(seq[:, 0:2], scale_factor=0.2, rot_thresh=45.0)

            if self.mode == 'Train':
               seq[:, 0:2] = SketchUtil.Q414k_horizontal_flip(seq[:, 0:2]/256)*256

            img = draw_three(seq, stroke_flag=0)
            seq[:, 0:2] = seq[:, 0:2] / 256

        else:
            index_neg = index_neg[0]
            seq[:index_neg,:] = off2abs(seq)[:index_neg,:]

            if random.uniform(0, 1) > 0.5 and self.mode == 'Train':
               seq[:, 0:2] = SketchUtil.random_affine_transform(seq[:, 0:2], scale_factor=0.2, rot_thresh=45.0)
            if self.mode == 'Train':
               seq[:index_neg, 0:2] = SketchUtil.Q414k_horizontal_flip(seq[:index_neg, 0:2]/256)*256

            img = draw_three(seq, stroke_flag=0)
            seq[:index_neg, 0:2] = seq[:index_neg, 0:2] / 256
        img_raw = Image.open(sketch_url, 'r')


        if self.mode == 'Train':
            sketch_img = self.train_transform(img)
            sketch_img_raw = self.train_transform(img_raw)
        elif self.mode == 'Test':
            sketch_img = self.test_transform(img)
            sketch_img_raw = self.test_transform(img_raw)
        elif self.mode == 'Valid':
            sketch_img = self.valid_transform(img)
            sketch_img_raw = self.valid_transform(img_raw)
        sample = {'sketch_img': sketch_img, 'sketch_points': seq, 'sketch_img_raw': sketch_img_raw,
                  'sketch_label': label, 'seq_len': 100}
        return sample


def collate_self(batch):
    batch_mod = {'sketch_img': [], 'sketch_points': [],
                 'sketch_label': [], 'seq_len': [],
                 }

    for i_batch in batch:
        batch_mod['sketch_img'].append(i_batch['sketch_img'])
        batch_mod['sketch_label'].append(i_batch['sketch_label'])
        batch_mod['seq_len'].append(i_batch['seq_len'])

        padded_sketch = -np.ones([hp.seq_len, 3], dtype=np.int16)  # 搴斾娇鐢╥nt16锛屼箣鍓嶄娇鐢ㄤ簡uint16
        padded_sketch[:i_batch['seq_len'], :] = i_batch['sketch_points']

        batch_mod['sketch_points'].append(torch.tensor(padded_sketch / 1.0))

    batch_mod['sketch_img'] = torch.stack(batch_mod['sketch_img'], dim=0)
    batch_mod['sketch_label'] = torch.tensor(batch_mod['sketch_label'])
    batch_mod['seq_len'] = torch.tensor(batch_mod['seq_len'])
    batch_mod['sketch_points'] = torch.stack(batch_mod['sketch_points'], dim=0).to(torch.float32)
    return batch_mod


def get_dataloader():
    if hp.Dataset == 'TUBerlin':

        dataset_Train = Dataset_TUBerlin(mode='Train')
        dataset_Test = Dataset_TUBerlin(mode='Test')
        dataset_Valid = Dataset_TUBerlin(mode='Valid')


    elif hp.Dataset == 'QuickDraw':

        dataset_Train = Dataset_Quickdraw(mode='Train')
        dataset_Test = Dataset_Quickdraw(mode='Test')
        dataset_Valid = Dataset_Quickdraw(mode='Valid')

    elif hp.Dataset == 'QuickDraw414k':
        dataset_Train = Quickdraw414k(mode='Train')
        dataset_Test = Quickdraw414k(mode='Test')
        dataset_Valid = Quickdraw414k(mode='Valid')

    dataloader_Train = data.DataLoader(dataset_Train, batch_size=hp.batchsize, shuffle=True, pin_memory=True,
                                       num_workers=int(hp.nThreads))

    dataloader_Test = data.DataLoader(dataset_Test, batch_size=1, shuffle=False,
                                      num_workers=int(hp.nThreads))

    dataloader_Valid = data.DataLoader(dataset_Valid, batch_size=300, shuffle=False,
                                       num_workers=int(hp.nThreads))
    # dataloader_Train = data.DataLoader(dataset_Train, batch_size=hp.batchsize, shuffle=True,
    #                                      num_workers=int(hp.nThreads), collate_fn=collate_self)
    #
    # dataloader_Test = data.DataLoader(dataset_Test, batch_size=hp.batchsize, shuffle=False,
    #                                      num_workers=int(hp.nThreads), collate_fn=collate_self)
    #
    # dataloader_Valid = data.DataLoader(dataset_Valid, batch_size=hp.batchsize, shuffle=False,
    #                                   num_workers=int(hp.nThreads), collate_fn=collate_self)

    return dataloader_Train, dataloader_Test, dataloader_Valid


def get_ransform(type):
    transform_list = []
    # if type is 'Train':
    # transform_list.extend([transforms.RandomRotation(45), transforms.RandomHorizontalFlip()])
    # elif type is 'Test':
    #     transform_list.extend([transforms.Resize(256)])
    # elif type is 'Valid':
    #     transform_list.extend([transforms.Resize(256)])
    # transform_list.extend(
    #     [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_list.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    return transforms.Compose(transform_list)
