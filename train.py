import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import argparse
import skimage
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
import numpy as np
from net.MultiTaskModel import RectanglingNetwork
from torch.utils.data import DataLoader
from utils.dataSet import SPRectanglingTrainDataSet,SPRectanglingTestDataSet
from net.loss_functions import MultiTask_loss
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from utils.learningRateScheduler import warmUpLearningRate
import random
from thop import profile

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)

setup_seed(2023)

def train_once(model,each_data_batch,epoch,epochs,criterion, optimizer):
    model.train()
    with tqdm(total=each_data_batch, desc=f'Epoch {epoch + 1}/{epochs}', postfix=dict, mininterval=0.3) as pbar:
        each_batch_all_loss = 0
        each_batch_primary_loss = 0
        each_batch_super_loss = 0
        print("Start Train")
        for i, (img1,mask,lables,super_gt) in enumerate(dataloders['train']):
            input_img = img1.float().cuda(device=args.device_ids[0])
            mask_img = mask.float().cuda(device=args.device_ids[0])
            lables = lables.float().cuda(device=args.device_ids[0])
            super_gt = super_gt.float().cuda(device=args.device_ids[0])

            optimizer.zero_grad()
            mesh_final, warp_image_final, warp_mask_final, super_image = model.forward(input_img, mask_img)

            loss,primary_img_loss,super_img_loss = criterion(mesh_final, warp_image_final, lables, super_image, super_gt)
            loss.backward()
            optimizer.step()

            each_batch_all_loss += loss.item() / args.train_batch_size
            each_batch_primary_loss += primary_img_loss.item() / args.train_batch_size
            each_batch_super_loss += super_img_loss.item() / args.train_batch_size
            pbar.set_postfix({'Lsum': each_batch_all_loss / (i + 1),
                              'Lpri': each_batch_primary_loss / (i + 1),
                              'Lsuper': each_batch_super_loss / (i + 1),
                              'lr': scheduler.get_last_lr()[0]})
            pbar.update(1)
        print("\nFinish Train")
        return each_batch_all_loss / each_data_batch

def val_once(model,t_each_data_batch,epoch,epochs,criterion,optimizer):
    model.eval()
    with tqdm(total=t_each_data_batch, desc=f'Epoch {epoch + 1}/{epochs}', postfix=dict, mininterval=0.3) as t_pbar:
        each_batch_psnr = 0
        each_batch_ssim = 0
        print("Start Test")
        with torch.no_grad():
            for i, (img1, mask, lables, super_img_in) in enumerate(dataloders['test']):
                input_img = img1.float().cuda(device=args.device_ids[0])
                mask_img = mask.float().cuda(device=args.device_ids[0])
                lables = lables.float().cuda(device=args.device_ids[0])
                super_img_in = super_img_in.float().cuda(device=args.device_ids[0])

                optimizer.zero_grad()
                mesh_final, warp_image_final, warp_mask_final, super_image = model.forward(input_img, mask_img)

                I1 = super_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
                I2 = super_img_in.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

                psnr = compare_psnr(I1, I2 , data_range=1)
                ssim = compare_ssim(I1 , I2, data_range=1, multichannel=True)

                each_batch_psnr += psnr / args.test_batch_size
                each_batch_ssim += ssim / args.test_batch_size
                t_pbar.set_postfix({'average psnr': each_batch_psnr / (i + 1),
                                    'average ssim': each_batch_ssim / (i + 1)})
                t_pbar.update(1)
        print("\nFinish Test")

def train(model,saveModelName,criterion,optimizer,scheduler,epochs=1):
    loss_history = []
    each_data_batch = len(dataloders['train'])
    t_each_data_batch = len(dataloders['test'])
    for epoch in range(epochs):
        # 训练
        # val_once(model, t_each_data_batch, epoch, epochs, criterion, optimizer)
        each_batch_all_loss = train_once(model,each_data_batch,epoch, epochs,criterion,optimizer)
        if epoch % 10 == 0:
            # 测试
            val_once(model, t_each_data_batch, epoch, epochs, criterion, optimizer)
        # learning rate scheduler
        scheduler.step()
        # print("epoch:",epoch,"lr:",scheduler.get_last_lr())
        loss_history.append(each_batch_all_loss)

        if (epoch + 1) % 10 ==0 or epoch >= int(epochs-5):
            torch.save(model.state_dict(), saveModelName + "_" + "epoch" + str(epoch + 1) + ".pkl")
            np.save(saveModelName + "_" + "epoch" + str(epoch + 1) + "_" + "TrainLoss" +
            str(round(each_batch_all_loss, 3)), np.array(loss_history))

    show_plot(loss_history)

def show_plot(loss_history):
    counter = range(len(loss_history))
    plt.plot(counter, loss_history)
    plt.legend(['train loss'])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=r'D:\dataResource\imageStitiching\DIR-D\DIR-D')
    parser.add_argument('--device_ids', type=list, default=[0])
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training') 
    parser.add_argument('--img_h', type=int, default=384)
    parser.add_argument('--img_w', type=int, default=512)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--save_model_name', type=str, default='./model/multiTask_model')
    parser.add_argument('--lam_perception', type=float, default=0.2)
    parser.add_argument('--lam_mask', type=float, default=1)
    parser.add_argument('--lam_mesh', type=float, default=1)
    parser.add_argument('--lam_appearance', type=float, default=1)
    print('<==================== Loading data ===================>\n')
    args = parser.parse_args()
    print(args)
    ##############

    pathGT = os.path.join(args.path, 'training/gt')
    pathInput = os.path.join(args.path, 'training/input')
    pathMask = os.path.join(args.path, 'training/mask')
    pathGT2 = os.path.join(args.path, 'testing/gt')
    pathInput2 = os.path.join(args.path, 'testing/input')
    pathMask2 = os.path.join(args.path, 'testing/mask')
    image_datasets = {}
    image_datasets['train'] = SPRectanglingTrainDataSet(pathInput, pathMask, pathGT, args.img_h, args.img_w)
    image_datasets['test'] = SPRectanglingTestDataSet(pathInput2, pathMask2, pathGT2, args.img_h, args.img_w)
    
    dataloders = {}
    # print("data:",next(iter(image_datasets['train'])))
    dataloders['train'] = DataLoader(image_datasets['train'], batch_size=args.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dataloders['test'] = DataLoader(image_datasets['test'], batch_size=args.test_batch_size, shuffle=False, num_workers=4)
    # define somethings
    criterion = MultiTask_loss(args.lam_appearance, args.lam_perception, args.lam_mask, args.lam_mesh).cuda(device=args.device_ids[0])
    # model = torch.nn.DataParallel(RectanglingNetwork(), device_ids=args.device_ids) # 指定要用到的设备
    model = RectanglingNetwork()
    # model = RectanglingNetwork(inference_mode=True)
    model = model.to(device=args.device_ids[0]) # 模型加载到设备0
    # ################
    re_img1 = torch.rand(1, 3, args.img_h, args.img_w).float().to(args.device_ids[0])
    mask1 = torch.rand(1, 3, args.img_h, args.img_w).float().to(args.device_ids[0])
    tensor = (re_img1, mask1)
    # 分析FLOPs
    flops, params = profile(model, inputs=tensor)
    print("Number of parameter: %.2f M" % (params / 1e6))
    print("Number of GFLOPs: %.2f GFLOPs" % (flops / 1e9))
    # ###############
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,weight_decay=1e-4)
    lrScheduler = warmUpLearningRate(args.max_epoch, warm_up_epochs=10, scheduler='cosine')
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lrScheduler)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True, weight_decay=1e-4)  # default as 0.0001
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.max_epoch, max_lr=args.lr, three_phase=True)
       
    # start train
    train(model, args.save_model_name, criterion, optimizer, scheduler, args.max_epoch)

