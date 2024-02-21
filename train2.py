import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import argparse
import skimage
from tqdm import tqdm
import numpy as np
from net.SwinAllmodel import RectanglingNetwork
from torch.utils.data import DataLoader
from utils.dataSet import RectanglingTrainDataSet,RectanglingTestDataSet
from net.loss_functions import Total_loss
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from utils.learningRateScheduler import warmUpLearningRate

def train_once(model,each_data_batch,epoch,epochs,criterion, optimizer):
    model.train()
    with tqdm(total=each_data_batch, desc=f'Epoch {epoch + 1}/{epochs}', postfix=dict, mininterval=0.3) as pbar:
        each_batch_all_loss = 0
        each_batch_appearance_loss = 0
        each_batch_perception_loss = 0
        each_batch_mask_loss = 0
        each_batch_mesh_loss = 0
        each_batch_super_loss = 0
        print("Start Train")
        for i, (img1,mask,lables,super_gt_img) in enumerate(dataloders['train']):
            input_img = img1.float().to(args.gpu_device)
            mask_img = mask.float().to(args.gpu_device)
            lables = lables.float().to(args.gpu_device)
            super_gt_img = super_gt_img.float().to(args.gpu_device)
            # import matplotlib.pyplot as plt
            # plt.subplot(131)
            # plt.imshow(input1[0,...].permute(1,2,0).cpu().detach().numpy())
            # plt.subplot(132)
            # plt.imshow(input2[0,...].permute(1,2,0).cpu().detach().numpy())
            # plt.subplot(133)
            # plt.imshow(lables[0,...].permute(1,2,0).cpu().detach().numpy())
            # plt.show()

            optimizer.zero_grad()
            mesh_primary, warp_image_primary, warp_mask_primary, super_image = model.forward(input_img, mask_img)
            # print(mesh_primary.shape)
            # print(warp_image_primary.shape)
            # print(warp_mask_primary.shape)
            loss,appearance_loss,perception_loss,mask_loss,mesh_loss,super_loss = criterion(mesh_primary, warp_image_primary, warp_mask_primary, lables,super_image,super_gt_img)
            loss.backward()
            optimizer.step()

            each_batch_all_loss += loss.item() / args.train_batch_size
            each_batch_appearance_loss += appearance_loss.item() / args.train_batch_size
            each_batch_perception_loss += perception_loss.item() / args.train_batch_size
            each_batch_mask_loss += mask_loss.item() / args.train_batch_size
            each_batch_mesh_loss += mesh_loss.item() / args.train_batch_size
            each_batch_super_loss += super_loss.item() / args.train_batch_size
            pbar.set_postfix({'Lsum': each_batch_all_loss / (i + 1),
                              'Lp': each_batch_appearance_loss / (i + 1),
                              'Lper': each_batch_perception_loss / (i + 1),
                              'Lsuper': each_batch_super_loss / (i + 1),
                              'Lmesh': each_batch_mesh_loss / (i + 1),
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
            for i, (img1, mask, lables,super_gt_img) in enumerate(dataloders['test']):
                input_img = img1.float().to(args.gpu_device)
                mask_img = mask.float().to(args.gpu_device)
                lables = lables.float().to(args.gpu_device)
                super_gt_img = super_gt_img.float().to(args.gpu_device)

                optimizer.zero_grad()
                mesh_primary, warp_image, warp_mask, super_image = model.forward(input_img, mask_img)

                I1 = super_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
                I2 = super_gt_img.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

                psnr = compare_psnr(I1, I2 , data_range=1)
                ssim = compare_ssim(I1 , I2, data_range=1, channel_axis=2)

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
        each_batch_all_loss = train_once(model,each_data_batch,epoch, epochs,criterion,optimizer)
        if epoch % 10 == 0:
            # 测试
            val_once(model, t_each_data_batch, epoch, epochs, criterion, optimizer)
        # learning rate scheduler
        scheduler.step()
        # print("epoch:",epoch,"lr:",scheduler.get_last_lr())
        loss_history.append(each_batch_all_loss)

        if (epoch + 1) >= 100:
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
    parser.add_argument('--gpu_device', type=str, default="cuda", help='Number of splits')
    parser.add_argument('--img_h', type=int, default=256)
    parser.add_argument('--img_w', type=int, default=256)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=120)
    parser.add_argument('--lr', type=float, default=4e-4, help='learning rate')
    parser.add_argument('--save_model_name', type=str, default='model/transformer_dimC8_mesh384512_model')
    parser.add_argument('--lam_perception', type=float, default=5e-6)
    parser.add_argument('--lam_mask', type=float, default=1)
    parser.add_argument('--lam_mesh', type=float, default=1)
    parser.add_argument('--lam_appearance', type=float, default=1)
    print('<==================== Loading data ===================>\n')
    args = parser.parse_args()
    print(args)
    ##############
    pathGT = os.path.join(args.path, 'training\gt')
    pathInput = os.path.join(args.path, 'training\input')
    pathMask = os.path.join(args.path, 'training\mask')
    pathGT2 = os.path.join(args.path, 'testing\gt')
    pathInput2 = os.path.join(args.path, 'testing\input')
    pathMask2 = os.path.join(args.path, 'testing\mask')
    image_datasets = {}
    image_datasets['train'] = RectanglingTrainDataSet(pathInput, pathMask, pathGT, args.img_h, args.img_w)
    image_datasets['test'] = RectanglingTestDataSet(pathInput2, pathMask2, pathGT2, args.img_h, args.img_w)
    dataloders = {}
    # print("data:",next(iter(image_datasets['train'])))
    dataloders['train'] = DataLoader(image_datasets['train'], batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    dataloders['test'] = DataLoader(image_datasets['test'], batch_size=args.test_batch_size, shuffle=False, num_workers=4)
    # define somethings
    criterion = Total_loss(args.lam_appearance, args.lam_perception, args.lam_mask, args.lam_mesh).to(args.gpu_device)
    model = RectanglingNetwork().to(args.gpu_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,weight_decay=1e-4)
    lrScheduler = warmUpLearningRate(args.max_epoch, warm_up_epochs=2, scheduler='cosine')
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lrScheduler)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True, weight_decay=1e-4)  # default as 0.0001
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    # start train
    train(model, args.save_model_name, criterion, optimizer, scheduler, args.max_epoch)
    # loss show
    # lossTotal = np.load('model/single_repconv_spp_model_dataAug_lr_weightMesh_epoch80_TrainLoss0.276_TestLoss0.536.npy')
    # print(lossTotal)
