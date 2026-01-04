import argparse
import os
import sys
import torch
import json
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

# model - resnet34
from Model.nets import resnet34, resnet18
from Model.nets import convnet

# 添加 SketchMLP 到路径以导入 Dataset
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'SketchMLP'))
from Dataset2 import Quickdraw414k

# 混合精度训练
try:
    from torch.cuda.amp import GradScaler, autocast
    AMP_AVAILABLE = True
    print("AMP (混合精度训练) 可用")
except ImportError:
    AMP_AVAILABLE = False
    print("AMP 不可用，使用普通训练")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch implementation of image classification based on Quick, Draw! data.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # parser.add_argument('--data_root', '-root', type=str, default='Dataset',
    #                     help='root for the dataset directory.')
    parser.add_argument('--image_size', '-size', type=int, default=224,
                        help='the size of the input image.')
    parser.add_argument('--in_channels', '-ic', type=int, default=3,
                        help='number of input channels (1 for grayscale, 3 for RGB).')
    parser.add_argument('--num_classes', '-nc', type=int, default=345,
                        help='number of classes for QuickDraw414k.')
    parser.add_argument('--raw_imags', '-ri', type=bool, default=False,
                        help='whether to use raw images.')

    # training
    parser.add_argument('--epochs', '-e', type=int, default=50,
                        help='number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int,
                        default=256, help='batch size.')
    parser.add_argument('--learning_rate', '-lr', type=float,
                        default=0.1, help='the learningrate.')
    parser.add_argument('--momentum', '-mo', type=float,
                        default=0.9, help='momentum.')
    parser.add_argument('--weight_decay', '-wd', type=float,
                        default=5e-4, help='L2 penalty weight decay.')
    parser.add_argument('--lr_decay_step', '-lrs',
                        type=int, nargs='*', default=[12, 20])
    parser.add_argument('--gamma', '-g', type=float, default=0.1,
                        help='lr is multiplied by gamma on step defined above.')
    parser.add_argument('--ngpu', type=int,
                        default=1, help='0 or less for CPU.')
    parser.add_argument('--model', '-m', type=str,
                        default='resnet18', help='choose the model.')
    parser.add_argument('--pretrained', type=bool, default=False, help='whether to use pretrained model.')

    # 余弦退火参数
    parser.add_argument('--scheduler', type=str, default='cosine', 
                       choices=['step', 'cosine', 'warm_restart'], 
                       help='learning rate scheduler type')
    parser.add_argument('--t_max', type=int, default=10,
                       help='T_max for CosineAnnealingLR')
    parser.add_argument('--t_0', type=int, default=10,
                       help='T_0 for CosineAnnealingWarmRestarts')
    parser.add_argument('--t_mult', type=int, default=2,
                       help='T_mult for CosineAnnealingWarmRestarts')
    parser.add_argument('--eta_min', type=float, default=1e-6,
                       help='minimum learning rate for cosine schedulers')

    # 混合精度训练
    parser.add_argument('--amp', action='store_true', default=False,
                       help='use mixed precision training')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='gradient clipping norm')

    # 损失函数参数
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                       help='label smoothing factor (0.0 to disable)')
    parser.add_argument('--focal_loss', action='store_true', default=False,
                       help='use focal loss instead of cross entropy')

    # 数据增强参数
    parser.add_argument('--augmentation', type=str, default='medium',
                       choices=['weak', 'medium', 'strong'],
                       help='data augmentation strength')

    # testing
    parser.add_argument('--test_bs', '-tb', type=int,
                        default=64, help='test batch size.')

    # checkpoint
    parser.add_argument('--save_dir', '-s', type=str,
                        default='./Checkpoints', help='directory for saving checkpoints')
    parser.add_argument('--save_freq', '-sf', type=int, default=1,
                        help='save checkpoint every N epochs.')
    parser.add_argument('--resume', '-r', type=str, default=None,
                        help='path to checkpoint to resume from.')

    # for log info
    parser.add_argument('--log', type=str, default='./',
                        help='path of the log info.')

    # 测试时增强
    parser.add_argument('--tta', action='store_true', default=False,
                       help='use test time augmentation')

    args = parser.parse_args()

    # 检查AMP可用性
    if args.amp and not AMP_AVAILABLE:
        print("警告：AMP不可用，将使用普通训练")
        args.amp = False

    device = torch.device('cuda' if torch.cuda.is_available() and args.ngpu > 0 else 'cpu')
    print(f"使用设备: {device}")
    print(f"学习率调度器: {args.scheduler}")
    print(f"数据增强强度: {args.augmentation}")
    if args.amp:
        print("启用混合精度训练")

    if not os.path.isdir(args.log):
        os.makedirs(args.log)

    log = open(os.path.join(args.log, 'log.txt'), 'w')
    state = {k: v for k, v in args._get_kwargs()}
    log.write(json.dumps(state)+'\n')

    # 初始化保存目录
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    print("*"*50)
    print("加载数据...")
    
    # 使用 SketchMLP 的 Quickdraw414k 数据集
    train_data = Quickdraw414k(mode="Train", augmentation=args.augmentation)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    test_data = Quickdraw414k(mode="Test", augmentation='weak')
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.test_bs, shuffle=False, num_workers=8, pin_memory=True)
    
    valid_data = Quickdraw414k(mode="Valid", augmentation='weak')
    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=args.test_bs, shuffle=False, num_workers=8, pin_memory=True)

    num_classes = args.num_classes

    print(f"训练图像数量: {len(train_data)}")
    print(f"测试图像数量: {len(test_data)}")
    print(f"验证图像数量: {len(valid_data)}")

    net = None
    if args.model == 'resnet34':
        net = resnet34(num_classes, pretrained=args.pretrained, in_channels=args.in_channels, image_size=args.image_size)
    elif args.model == 'resnet18':
        print('----------------18----------')
        net = resnet18(num_classes, pretrained=args.pretrained, in_channels=args.in_channels, image_size=args.image_size)
    elif args.model == 'convnet':
        net = convnet(num_classes)
    
    net = net.to(device)
    
    if args.ngpu > 1 and torch.cuda.device_count() > 1:
        print(f"=> 使用 {torch.cuda.device_count()} 个GPU进行数据并行")
        net = nn.DataParallel(net)

    if args.ngpu > 0:
        # 确保CUDA可用
        assert torch.cuda.is_available(), "CUDA不可用。请检查GPU和PyTorch安装。"
        # 如果需要，可以指定具体哪张卡，例如 `cuda:0`
        torch.cuda.set_device(0)
        print(f"=> 使用GPU: {torch.cuda.get_device_name(0)}")

    print(net)

    # 优化器
    optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'],
                                momentum=state['momentum'], weight_decay=state['weight_decay'])
    
    # 学习率调度器
    scheduler = None
    if args.scheduler == 'step':
        # 原来的步长衰减
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif args.scheduler == 'cosine':
        # 余弦退火
        scheduler = CosineAnnealingLR(optimizer, T_max=args.t_max, eta_min=args.eta_min)
    elif args.scheduler == 'warm_restart':
        # 余弦退火热重启
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.t_0, T_mult=args.t_mult, eta_min=args.eta_min)
    
    # 混合精度训练的梯度缩放器
    scaler = GradScaler() if args.amp else None

    # 损失函数
    if args.focal_loss:
        # Focal Loss实现
        class FocalLoss(nn.Module):
            def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
                super(FocalLoss, self).__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.reduction = reduction
            
            def forward(self, inputs, targets):
                ce_loss = F.cross_entropy(inputs, targets, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                
                if self.reduction == 'mean':
                    return focal_loss.mean()
                elif self.reduction == 'sum':
                    return focal_loss.sum()
                else:
                    return focal_loss
        
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        print("使用Focal Loss")
    elif args.label_smoothing > 0:
        # 标签平滑交叉熵
        class LabelSmoothingCrossEntropy(nn.Module):
            def __init__(self, smoothing=0.1):
                super(LabelSmoothingCrossEntropy, self).__init__()
                self.smoothing = smoothing
            
            def forward(self, pred, target):
                log_probs = F.log_softmax(pred, dim=-1)
                nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
                nll_loss = nll_loss.squeeze(1)
                smooth_loss = -log_probs.mean(dim=-1)
                loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
                return loss.mean()
        
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
        print(f"使用标签平滑交叉熵 (smoothing={args.label_smoothing})")
    else:
        criterion = nn.CrossEntropyLoss()
        print("使用标准交叉熵损失")

    # 从检查点恢复训练
    start_epoch = 0
    best_accuracy = 0.0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> 加载检查点 '{args.resume}'")
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_accuracy = checkpoint.get('best_accuracy', 0.0)
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            state['learning_rate'] = checkpoint.get('learning_rate', state['learning_rate'])
            print(f"=> 从epoch {start_epoch}恢复，最佳准确率: {best_accuracy:.4f}")
        else:
            print(f"=> 在 '{args.resume}' 找不到检查点")

    def train():
        net.train()
        loss_avg = 0.0
        correct = 0
        total = 0
        
        # 信息在终端打印
        data_loader = tqdm(train_loader, desc='训练')
        for batch_idx, batch in enumerate(data_loader):
            # 适配 Quickdraw414k 数据集的字典格式
            if args.raw_imags:
                data = batch['sketch_img_raw']
            else:
                data = batch['sketch_img']
            target = batch['sketch_label']
            
            if args.ngpu > 0:
                data, target = data.cuda(), target.cuda()

            # 前向传播
            optimizer.zero_grad()
            
            if args.amp:
                # 混合精度训练
                with autocast():
                    output = net(data)
                    loss = criterion(output, target)
                
                # 反向传播
                scaler.scale(loss).backward()
                
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
                
                # 优化器步进
                scaler.step(optimizer)
                scaler.update()
            else:
                # 普通训练
                output = net(data)
                loss = criterion(output, target)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
                
                # 优化器步进
                optimizer.step()

            # 准确率计算
            pred = output.data.max(1)[1]
            correct += float(pred.eq(target.data).sum())
            total += target.size(0)

            # 指数移动平均损失
            loss_avg = loss_avg * 0.9 + float(loss) * 0.1
            
            # 更新进度条
            data_loader.set_postfix({
                'loss': f'{loss_avg:.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        state['train_loss'] = loss_avg
        state['train_accuracy'] = correct / len(train_loader.dataset)

    def test(use_tta=False):
        net.eval()
        loss_avg = 0.0
        correct = 0
        
        # 测试时增强函数
        def tta_predict(data, n_aug=3):
            """测试时增强预测"""
            outputs = []
            
            # 原始图像
            with torch.no_grad():
                output = net(data)
                outputs.append(F.softmax(output, dim=1))
            
            # 数据增强预测
            if use_tta:
                for i in range(n_aug):
                    # 随机水平翻转
                    if i % 2 == 0:
                        aug_data = torch.flip(data, dims=[3])
                    else:
                        aug_data = data
                    
                    # 随机小角度旋转（简化版本，避免复杂变换）
                    if i % 3 == 0:
                        # 使用简单的旋转
                        angle = np.random.uniform(-10, 10)
                        # 创建旋转矩阵
                        theta = torch.tensor([
                            [np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle)), 0],
                            [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle)), 0]
                        ], dtype=torch.float32, device=data.device).unsqueeze(0)
                        # 创建网格
                        grid = F.affine_grid(theta, aug_data.size(), align_corners=False)
                        aug_data = F.grid_sample(aug_data, grid, align_corners=False)
                    
                    with torch.no_grad():
                        output = net(aug_data)
                        outputs.append(F.softmax(output, dim=1))
            
            # 平均预测
            final_output = torch.stack(outputs).mean(0)
            return final_output
        
        # 信息在终端打印
        data_loader = tqdm(test_loader, desc='测试(TTA)' if use_tta else '测试')
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                # 适配 Quickdraw414k 数据集的字典格式
                if args.raw_imags:
                    data = batch['sketch_img_raw']
                else:
                    data = batch['sketch_img']
                target = batch['sketch_label']
                
                if args.ngpu > 0:
                    data, target = data.cuda(), target.cuda()

                # 前向传播
                if use_tta:
                    output = tta_predict(data, n_aug=3)
                else:
                    output = net(data)
                
                loss = criterion(output, target)

                # 准确率
                pred = output.data.max(1)[1]
                correct += float(pred.eq(target.data).sum())

                # 测试损失平均
                loss_avg += float(loss)
                
                # 更新进度条
                data_loader.set_postfix({
                    'acc': f'{100.*correct/((batch_idx+1)*data.size(0)):.2f}%'
                })

        state['test_loss'] = loss_avg / len(test_loader)
        state['test_accuracy'] = correct / len(test_loader.dataset)

    # 主循环
    for epoch in range(start_epoch, args.epochs):
        print("")
        print("*"*50)
        print(f"epoch {epoch+1}/{args.epochs} 正在运行...")
        
        # 更新学习率调度器
        if args.scheduler == 'step':
            if epoch+1 in args.lr_decay_step:
                state['learning_rate'] *= args.gamma
                for param_group in optimizer.param_groups:
                    param_group['lr'] = state['learning_rate']
        elif scheduler is not None:
            scheduler.step()
            state['learning_rate'] = optimizer.param_groups[0]['lr']

        state['current_epoch'] = epoch+1
        state['current_lr'] = state['learning_rate']
        
        # 训练和测试
        train()
        print("")
        
        # 测试（可选使用TTA）
        if args.tta and (epoch + 1) % 5 == 0:  # 每5个epoch使用一次TTA
            test(use_tta=True)
        else:
            test()
        
        print("")
        
        # 保存最佳模型
        if state['test_accuracy'] > best_accuracy:
            best_accuracy = state['test_accuracy']
            best_model_path = os.path.join(args.save_dir, 'model_best.pytorch')
            torch.save(net.state_dict(), best_model_path)
            print(f"=> 保存最佳模型到 '{best_model_path}'，准确率: {best_accuracy:.4f}")
        
        # 阶段性保存完整检查点（每 save_freq 个 epoch）
        if (epoch + 1) % args.save_freq == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_accuracy': best_accuracy,
                'learning_rate': state['learning_rate'],
            }
            ckpt_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, ckpt_path)
            print(f"=> 保存检查点到 '{ckpt_path}'")
        
        log.write('%s\n' % json.dumps(state))
        log.flush()
        print(f"当前状态: {state}")
        print(f"最佳准确率: {best_accuracy:.4f}")
        print(f"当前学习率: {state['learning_rate']:.6f}")
        print("*"*50)

    # 训练结束保存最终检查点
    final_checkpoint = {
        'epoch': args.epochs,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_accuracy': best_accuracy,
        'learning_rate': state['learning_rate'],
    }
    torch.save(final_checkpoint, os.path.join(args.save_dir, 'checkpoint_final.pth'))
    print("=> 保存最终检查点")
    
    # 最终评估
    print("\n最终评估:")
    test(use_tta=args.tta)
    print(f"最终测试准确率: {state['test_accuracy']:.4f}")
    print(f"训练期间最佳准确率: {best_accuracy:.4f}")
    
    log.close()