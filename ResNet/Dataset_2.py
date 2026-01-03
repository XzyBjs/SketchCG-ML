import os
import lmdb
import torch

import Utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pickle
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import numpy as np
from Utils import draw_three, off2abs
from Hyper_params import hp
from SketchUtils import SketchUtil
from PIL import Image

class Quickdraw414k(data.Dataset):

    def __init__(self, mode='Train', augmentation='medium'):
        self.mode = mode
        self.augmentation = augmentation
        
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
        print(f'总 {mode} 样本数: {len(self.labels)}')
        print(f'数据增强强度: {augmentation}')

        # 获取对应的数据增强
        if self.mode == 'Train':
            self.transform = get_transform(augmentation, mode='train')
        else:
            self.transform = get_transform('weak', mode='test')

    def __len__(self):
        return len(self.img_urls)

    def __getitem__(self, item):
        sketch_url = self.img_urls[item]
        coordinate_url = self.coordinate_urls[item]
        label = self.labels[item]

        # 加载坐标序列
        seq = np.load(coordinate_url, encoding='latin1', allow_pickle=True)
        if seq.dtype == 'object':
            seq = seq[0]
        assert seq.shape == (100, 4)
        seq = seq.astype('float32')
        seq = seq[:, 0:3]
        index_neg = np.where(seq == -1)[0]
        
        # 处理坐标并生成图像
        if len(index_neg) == 0:
            seq = off2abs(seq)

            # 训练时应用坐标增强
            if self.mode == 'Train':
                if random.uniform(0, 1) > 0.5:
                    seq[:, 0:2] = SketchUtil.random_affine_transform(
                        seq[:, 0:2], scale_factor=0.2, rot_thresh=45.0)
                
                # 随机水平翻转
                if random.uniform(0, 1) > 0.5:
                    seq[:, 0:2] = SketchUtil.Q414k_horizontal_flip(seq[:, 0:2]/256)*256

            # 从坐标生成图像
            img = draw_three(seq, stroke_flag=0)
            seq[:, 0:2] = seq[:, 0:2] / 256

        else:
            index_neg = index_neg[0]
            seq[:index_neg,:] = off2abs(seq)[:index_neg,:]

            # 训练时应用坐标增强
            if self.mode == 'Train':
                if random.uniform(0, 1) > 0.5:
                    seq[:, 0:2] = SketchUtil.random_affine_transform(
                        seq[:, 0:2], scale_factor=0.2, rot_thresh=45.0)
                
                # 随机水平翻转
                if random.uniform(0, 1) > 0.5:
                    seq[:index_neg, 0:2] = SketchUtil.Q414k_horizontal_flip(seq[:index_neg, 0:2]/256)*256

            # 从坐标生成图像
            img = draw_three(seq, stroke_flag=0)
            seq[:index_neg, 0:2] = seq[:index_neg, 0:2] / 256
        
        # 加载原始图像
        img_raw = Image.open(sketch_url, 'r')
        
        # 转换为PIL图像用于增强
        if isinstance(img, np.ndarray):
            # 将numpy数组转换为PIL图像
            if img.max() <= 1.0:
                img_pil = Image.fromarray((img * 255).astype(np.uint8))
            else:
                img_pil = Image.fromarray(img.astype(np.uint8))
        else:
            img_pil = img
        
        # 应用数据增强
        sketch_img = self.transform(img_pil)
        sketch_img_raw = self.transform(img_raw)

        sample = {
            'sketch_img': sketch_img, 
            'sketch_points': seq, 
            'sketch_img_raw': sketch_img_raw,
            'sketch_label': label, 
            'seq_len': 100
        }
        return sample


def get_transform(augmentation='medium', mode='train'):
    """
    获取数据增强转换
    augmentation: 'weak', 'medium', 'strong'
    mode: 'train', 'test', 'valid'
    """
    
    if mode != 'train':
        # 测试和验证模式使用弱增强
        transform_list = [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        return transforms.Compose(transform_list)
    
    # 训练模式根据强度选择增强
    if augmentation == 'weak':
        transform_list = [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    
    elif augmentation == 'medium':
        transform_list = [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3))
        ]
    
    elif augmentation == 'strong':
        # 尝试导入Albumentations
        try:
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
            
            # 检查Albumentations版本并调整参数格式
            import albumentations
            alb_version = albumentations.__version__
            print(f"使用Albumentations版本: {alb_version}")
            
            # 根据版本调整参数
            if int(alb_version.split('.')[0]) >= 1:
                # 新版本Albumentations
                transform_strong = A.Compose([
                    A.Resize(height=256, width=256),
                    A.RandomResizedCrop(height=224, width=224, scale=(0.7, 1.0)),
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=30, p=0.7),
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.7),
                    A.OneOf([
                        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
                        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
                    ], p=0.5),
                    A.OneOf([
                        A.GaussNoise(var_limit=(10.0, 50.0)),
                        A.GaussianBlur(blur_limit=(3, 7)),
                        A.MotionBlur(blur_limit=(3, 7)),
                    ], p=0.3),
                    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, fill_value=0, p=0.2),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
                
                # 创建自定义转换函数
                class AlbumentationsTransform:
                    def __init__(self, transform):
                        self.transform = transform
                    
                    def __call__(self, img):
                        # 将PIL图像转换为numpy数组
                        img_np = np.array(img)
                        # 应用Albumentations转换
                        augmented = self.transform(image=img_np)
                        return augmented['image']
                
                return AlbumentationsTransform(transform_strong)
            else:
                # 旧版本Albumentations
                raise ImportError("Albumentations版本过低，请升级到1.0+版本")
                
        except ImportError:
            print("警告: Albumentations未安装或版本过低，使用torchvision强增强")
            # 使用torchvision的强增强作为备选
            transform_list = [
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(30),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
                transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3))
            ]
            return transforms.Compose(transform_list)
    
    # 对于weak和medium，直接返回torchvision转换
    return transforms.Compose(transform_list)


def collate_self(batch):
    batch_mod = {
        'sketch_img': [], 
        'sketch_points': [],
        'sketch_label': [], 
        'seq_len': [],
    }

    for i_batch in batch:
        batch_mod['sketch_img'].append(i_batch['sketch_img'])
        batch_mod['sketch_label'].append(i_batch['sketch_label'])
        batch_mod['seq_len'].append(i_batch['seq_len'])

        padded_sketch = -np.ones([hp.seq_len, 3], dtype=np.int16)
        padded_sketch[:i_batch['seq_len'], :] = i_batch['sketch_points']

        batch_mod['sketch_points'].append(torch.tensor(padded_sketch / 1.0))

    batch_mod['sketch_img'] = torch.stack(batch_mod['sketch_img'], dim=0)
    batch_mod['sketch_label'] = torch.tensor(batch_mod['sketch_label'])
    batch_mod['seq_len'] = torch.tensor(batch_mod['seq_len'])
    batch_mod['sketch_points'] = torch.stack(batch_mod['sketch_points'], dim=0).to(torch.float32)
    return batch_mod


def get_dataloader(augmentation='medium'):
    if hp.Dataset == 'TUBerlin':
        dataset_Train = Dataset_TUBerlin(mode='Train')
        dataset_Test = Dataset_TUBerlin(mode='Test')
        dataset_Valid = Dataset_TUBerlin(mode='Valid')
    elif hp.Dataset == 'QuickDraw':
        dataset_Train = Dataset_Quickdraw(mode='Train')
        dataset_Test = Dataset_Quickdraw(mode='Test')
        dataset_Valid = Dataset_Quickdraw(mode='Valid')
    elif hp.Dataset == 'QuickDraw414k':
        dataset_Train = Quickdraw414k(mode='Train', augmentation=augmentation)
        dataset_Test = Quickdraw414k(mode='Test', augmentation='weak')
        dataset_Valid = Quickdraw414k(mode='Valid', augmentation='weak')

    dataloader_Train = data.DataLoader(
        dataset_Train, 
        batch_size=hp.batchsize, 
        shuffle=True, 
        pin_memory=True,
        num_workers=int(hp.nThreads)
    )

    dataloader_Test = data.DataLoader(
        dataset_Test, 
        batch_size=1, 
        shuffle=False,
        num_workers=int(hp.nThreads)
    )

    dataloader_Valid = data.DataLoader(
        dataset_Valid, 
        batch_size=300, 
        shuffle=False,
        num_workers=int(hp.nThreads)
    )
    
    return dataloader_Train, dataloader_Test, dataloader_Valid