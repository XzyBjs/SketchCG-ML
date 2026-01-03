import argparse
import os
import sys
import torch
import json
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# model - resnet34
from Model.nets import resnet34, resnet18
from Model.nets import convnet

# 添加 SketchMLP 到路径以导入 Dataset
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'SketchMLP'))
from Dataset import Quickdraw414k



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
    parser.add_argument('--epochs', '-e', type=int, default=30,
                        help='number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int,
                        default=256, help='batch size.')
    parser.add_argument('--learning_rate', '-lr', type=float,
                        default=0.0001, help='the learningrate.')
    parser.add_argument('--momentum', '-mo', type=float,
                        default=0.9, help='momentum.')
    parser.add_argument('--weight_decay', '-wd', type=float,
                        default=5e-4, help='L2 penalty weight decay.')
    # parser.add_argument('--lr_decay_step', '-lrs',
    #                     type=int, nargs='*', default=[12, 20])
    # parser.add_argument('--gamma', '-g', type=float, default=0.1,
    #                     help='lr is multiplied by gamma on step defined above.')
    parser.add_argument('--ngpu', type=int,
                        default=1, help='0 or less for CPU.')
    parser.add_argument('--model', '-m', type=str,
                        default='resnet18', help='choose the model.')
    parser.add_argument('--pretrained', type=bool, default=False, help='whether to use pretrained model.')

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

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and args.ngpu > 0 else 'cpu')
    print(f"Using device: {device}")

    if not os.path.isdir(args.log):
        os.makedirs(args.log)
    
    log = open(os.path.join(args.log, 'log.txt'), 'w')
    state = {k: v for k, v in args._get_kwargs()}
    log.write(json.dumps(state)+'\n')

    # Init save directory
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    print("*"*50)
    print("Loading the data...")
    
    # 使用 SketchMLP 的 Quickdraw414k 数据集
    train_data = Quickdraw414k(mode="Train")
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    test_data = Quickdraw414k(mode="Test")
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.test_bs, shuffle=False, num_workers=4, pin_memory=True)
    
    valid_data = Quickdraw414k(mode="Valid")
    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=args.test_bs, shuffle=False, num_workers=4, pin_memory=True)

    num_classes = args.num_classes

    print("Train images number: %d" % len(train_data))
    print("Test images number: %d" % len(test_data))

    net = None
    if args.model == 'resnet34':
        net = resnet34(num_classes, pretrained=args.pretrained, in_channels=args.in_channels, image_size=args.image_size)
    elif args.model == 'resnet18':
        net = resnet18(num_classes, pretrained=args.pretrained, in_channels=args.in_channels, image_size=args.image_size)
    elif args.model == 'convnet':
        net = convnet(num_classes)
    
    net = net.to(device)
    
    if args.ngpu > 1 and torch.cuda.device_count() > 1:
        print(f"=> Using {torch.cuda.device_count()} GPUs for DataParallel")
        net = nn.DataParallel(net)

    if args.ngpu > 0:
            # 确保CUDA可用
        assert torch.cuda.is_available(), "CUDA is not available. Check your GPU and PyTorch installation."
        # 如果需要，可以指定具体哪张卡，例如 `cuda:0`
        torch.cuda.set_device(0)
        print(f"=> Using GPU: {torch.cuda.get_device_name(0)}")

    print(net)

    optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'],
                                momentum=state['momentum'], weight_decay=state['weight_decay'])

    # 从 checkpoint 恢复训练
    start_epoch = 0
    best_accuracy = 0.0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_accuracy = checkpoint.get('best_accuracy', 0.0)
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            state['learning_rate'] = checkpoint.get('learning_rate', state['learning_rate'])
            print(f"=> Resumed from epoch {start_epoch}, best_accuracy: {best_accuracy:.4f}")
        else:
            print(f"=> No checkpoint found at '{args.resume}'")

    def train():
        net.train()
        loss_avg = 0.0
        correct = 0
        # info printed in terminal
        data_loader = tqdm(train_loader, desc='Training')
        for batch_idx, batch in enumerate(data_loader):
            # 适配 Quickdraw414k 数据集的字典格式
            if args.raw_imags:
                data = batch['sketch_img_raw']
            else:
                data = batch['sketch_img']
            target = batch['sketch_label']
            
            if args.ngpu > 0:
                data, target = data.cuda(), target.cuda()

            # Quickdraw414k 返回的已经是正确格式的图像，不需要额外处理
            # forward
            output = net(data)

            # backward
            optimizer.zero_grad()
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            # accuracy
            pred = output.data.max(1)[1]
            correct += float(pred.eq(target.data).sum())

            # exponential moving average
            loss_avg = loss_avg*0.2+float(loss)*0.8

        state['train_loss'] = loss_avg
        state['train_accuracy'] = correct/len(train_loader.dataset)

    def test():
        net.eval()
        loss_avg = 0.0
        correct = 0
        # info printed in terminal
        data_loader = tqdm(test_loader, desc='Testing')
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

                # forward
                output = net(data)
                loss = F.cross_entropy(output, target)

                # accuracy
                pred = output.data.max(1)[1]
                correct += float(pred.eq(target.data).sum())

                # test loss average
                loss_avg += float(loss)

        state['test_loss'] = loss_avg/len(test_loader)
        state['test_accuracy'] = correct/len(test_loader.dataset)

    # Main loop
    for epoch in range(start_epoch, args.epochs):
        print("")
        print("*"*50)
        print("epoch "+str(epoch+1)+" is running...")

        
        # if epoch+1 in args.lr_decay_step:
        #     state['learning_rate'] *= args.gamma
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = state['learning_rate']
        

        state['current_epoch'] = epoch+1

        # 打印完整的state信息
        print("Current state:")
        for key, value in state.items():
            print(f"  {key}: {value}")
        print("-"*30)
        # 在训练前打印当前学习率（确保打印）
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Training with learning rate: {current_lr}")
        sys.stdout.flush()  # 强制刷新输出缓冲区

        train()
        print("")
        test()
        print("")
        
        

        # 保存最佳模型
        if state['test_accuracy'] > best_accuracy:
            best_accuracy = state['test_accuracy']
            torch.save(net.state_dict(), os.path.join(
                args.save_dir, 'model_best.pytorch'))
        
        # 阶段性保存完整 checkpoint（每 save_freq 个 epoch）
        if (epoch + 1) % args.save_freq == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_accuracy,
                'learning_rate': state['learning_rate'],
            }
            ckpt_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, ckpt_path)
            print(f"=> Saved checkpoint to '{ckpt_path}'")
        
        log.write('%s\n' % json.dumps(state))
        log.flush()
        print(state)
        print("Best accuracy: %.4f" % best_accuracy)
        print("*"*50)

    # 训练结束保存最终 checkpoint
    final_checkpoint = {
        'epoch': args.epochs,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_accuracy': best_accuracy,
        'learning_rate': state['learning_rate'],
    }
    torch.save(final_checkpoint, os.path.join(args.save_dir, 'checkpoint_final.pth'))
    print("=> Saved final checkpoint")
    
    log.close()
