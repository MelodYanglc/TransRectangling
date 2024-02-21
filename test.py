import os

import skimage.color

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import cv2
import argparse
import torch
from net.SwinAllmodel import RectanglingNetwork
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def inference_func(pathInput2,pathMask2,pathGT2,model_path):
    # _origin_transform = transforms.Compose([
    #     transforms.ToTensor()
    # ])
    resize_w, resize_h = args.img_w,args.img_h
    _origin_transform = transforms.Compose([
        transforms.Resize([resize_h, resize_w]),
        transforms.ToTensor(),
    ])
    _origin_transform2 = transforms.Compose([
        transforms.Resize([384, 512]),
        transforms.ToTensor(),
    ])
    index_all = list(sorted([x.split('.')[0] for x in os.listdir(pathInput2)]))
    # load model
    model = RectanglingNetwork()
    model.load_state_dict(torch.load(model_path))
    model = model.to(args.gpu_device)
    # model.featureExtrator.fuse()
    # model.meshRegression.fuse()
    # print(model)
    model.eval()
    psnr_list = []
    ssim_list = []
    length = 519
    for i in range(0, length):
        idx = index_all[i]
        # input_img = cv2.imread(os.path.join(pathInput2, str(idx) + '.jpg')) / 255.
        # input_img = cv2.resize(input_img, (resize_w, resize_h))
        # mask_img = cv2.imread(os.path.join(pathMask2, str(idx) + '.jpg')) / 255.
        # mask_img = cv2.resize(mask_img, (resize_w, resize_h))
        # gt_img = cv2.imread(os.path.join(pathGT2, str(idx) + '.jpg')) / 255.
        # gt_img = cv2.resize(gt_img, (resize_w, resize_h))
        # super_gt_img = cv2.imread(os.path.join(pathGT2, str(idx) + '.jpg')) / 255.
        # super_gt_img = cv2.resize(super_gt_img, (512,384))
        #
        # test_gt = _origin_transform(gt_img).unsqueeze(0).float().to(args.gpu_device)
        # test_input = _origin_transform(input_img).unsqueeze(0).float().to(args.gpu_device)
        # test_mask = _origin_transform(mask_img).unsqueeze(0).float().to(args.gpu_device)
        # super_gt_img = _origin_transform(super_gt_img).unsqueeze(0).float().to(args.gpu_device)

        input_img = cv2.imread(os.path.join(pathInput2, str(idx) + '.jpg'))
        mask_img = cv2.imread(os.path.join(pathMask2, str(idx) + '.jpg'))
        gt_img = cv2.imread(os.path.join(pathGT2, str(idx) + '.jpg'))
        from PIL import Image
        input_img = Image.fromarray(input_img)
        mask_img = Image.fromarray(mask_img)
        gt_img = Image.fromarray(gt_img)
        super_gt_img = gt_img.copy()
        ###
        test_input = _origin_transform(input_img).unsqueeze(0).float().to(args.gpu_device)
        test_mask = _origin_transform(mask_img).unsqueeze(0).float().to(args.gpu_device)
        test_gt = _origin_transform(gt_img).unsqueeze(0).float().to(args.gpu_device)
        # super_input_img = _origin_transform2(input_img).unsqueeze(0).float().to(args.gpu_device)
        super_gt_img = _origin_transform2(super_gt_img).unsqueeze(0).float().to(args.gpu_device)


        mesh_primary,  warp_image_primary, warp_mask_primary, super_image = model(test_input, test_mask)
        # import torch.nn as nn
        # warp_image_primary = nn.Upsample(size=(384, 512), mode='bilinear')(warp_image_primary)
        warp_image_primary = warp_image_primary.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        warp_image = super_image.squeeze(0).permute(1,2,0).cpu().detach().numpy()
        test_gt = super_gt_img.squeeze(0).permute(1,2,0).cpu().detach().numpy()
        # test_input = test_input.squeeze(0).permute(1,2,0).cpu().detach().numpy()
        # print("warp_image:",warp_image.min())
        # print("test_gt:", test_gt.shape)
        # import matplotlib.pyplot as plt
        # plt.subplot(121)
        # plt.imshow(warp_image,cmap='gray')
        # plt.subplot(122)
        # plt.imshow(test_input,cmap='gray')
        # plt.show()
        I1 = warp_image
        I2 = test_gt
        psnr = compare_psnr(I1, I2, data_range=1)
        ssim = compare_ssim(I1, I2, data_range=1, channel_axis=2)

        path = "final_rectangling/" + str(i + 1).zfill(5) + ".jpg"
        print(warp_image.shape)
        # warp_image = cv2.resize(warp_image, (512, 384))
        cv2.imwrite(path, warp_image_primary*255.)
        print('i = {} / {}, psnr = {:.6f}, ssim = {:.6f}'.format(i + 1, length, psnr,ssim))
        psnr_list.append(psnr)
        ssim_list.append(ssim)

    print("===================Results Analysis==================")
    print('average psnr:', np.mean(psnr_list))
    print('average ssim:', np.mean(ssim_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=r'D:\dataResource\imageStitiching\DIR-D\DIR-D')
    parser.add_argument('--gpu_device', type=str, default="cuda", help='Number of splits')
    parser.add_argument('--img_w', type=int, default=256)
    parser.add_argument('--img_h', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--save_model_name', type=str, default='model07726/transformer_dimC8_mesh384512_model_epoch120.pkl')
    parser.add_argument('--lam_perception', type=float, default=5e-6)
    parser.add_argument('--lam_mask', type=float, default=1)
    parser.add_argument('--lam_mesh', type=float, default=1)
    parser.add_argument('--lam_appearance', type=float, default=1)
    print('<==================== Loading data ===================>\n')

    args = parser.parse_args()
    print(args)
    ##############
    pathGT2 = os.path.join(args.path, 'testing\gt')
    pathInput2 = os.path.join(args.path, 'testing\input')
    pathMask2 = os.path.join(args.path, 'testing\mask')
    model_path = args.save_model_name
    # test
    inference_func(pathInput2,pathMask2,pathGT2,model_path)

        
        
        


    






