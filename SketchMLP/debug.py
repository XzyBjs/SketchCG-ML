from Dataset import get_dataloader
import torchvision.utils as vutils
import torch
import numpy as np

if __name__ == '__main__':

    print('''***********- debug -*************''')  
    dataloader_Train, dataloader_Test, dataloader_Valid = get_dataloader()
    it = iter(dataloader_Test)
    for i in range(5):
        batch = next(it)
    
    print("Batch keys:", batch.keys())
    print("Image shape:", batch['sketch_img'].shape)
    print("Labels:", batch['sketch_label'])
    
    # 取出图片 tensor，反归一化（原 normalize: mean=0.5, std=0.5）
    images = batch['sketch_img']
    images = images * 0.5 + 0.5  # 反归一化到 [0, 1]
    images = torch.clamp(images, 0, 1)
    
    # 保存前几张图片为网格
    vutils.save_image(images[:16], 'batch_samples.png', nrow=4, padding=2)
    print("已保存 batch_samples.png")

    images = batch['sketch_img_raw']
    images = images * 0.5 + 0.5  # 反归一化到 [0, 1]
    images = torch.clamp(images, 0, 1)
    
    # 保存前几张图片为网格
    vutils.save_image(images[:16], 'batch_samples2.png', nrow=4, padding=2)
    print("已保存 batch_samples.png")