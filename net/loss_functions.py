import torch
import numpy as np
import torch.nn as nn
from math import exp
import torch.nn.functional as F
from torchvision.models import vgg16, vgg19
from torch.autograd import Variable

import utils.constant as constant

grid_w = constant.GRID_W
grid_h = constant.GRID_H
gpu_device = constant.GPU_DEVICE

min_w = (512 / grid_w) / 8
min_h = (384 / grid_h) / 8
# min_w = (256 / grid_w) / 8
# min_h = (256 / grid_h) / 8


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class TV_Loss(torch.nn.Module):

    def __init__(self):
        super(TV_Loss, self).__init__()

    def forward(self, IA, IF):
        r = IA - IF
        h = r.shape[2]
        w = r.shape[3]
        tv1 = torch.pow((r[:, :, 1:, :] - r[:, :, :h - 1, :]), 2).mean()
        tv2 = torch.pow((r[:, :, :, 1:] - r[:, :, :, :w - 1]), 2).mean()

        return tv1 + tv2


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            window = window.to(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

class GradLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,img1,img2):
        device = img1.device
        b, c, h, w = img1.shape
        kernel = torch.FloatTensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) \
            .to(gpu_device)#.cuda().reshape((1, 1, 3, 3)).repeat(c, 1, 1, 1)
        f1_grad =  F.conv2d(img1, kernel,padding=1,groups=c)
        f2_grad =  F.conv2d(img2, kernel,padding=1,groups=c)
        totalGradLoss = intensity_loss(gen_frames=f1_grad, gt_frames=f2_grad, l_num=2)
        return totalGradLoss

class VGG(nn.Module):
    def __init__(self, layer_indexs):
        super(VGG, self).__init__()
        layers = []
        in_dim = 3
        out_dim = 64
        self.layer_indexs = layer_indexs
        # 循环构造卷积层，一共有13个卷积层
        for i in range(16):
            layers += [nn.Conv2d(in_dim, out_dim, 3, 1, 1), nn.ReLU(inplace=True)]
            in_dim = out_dim
            # 在第2、4、7、10、13个卷积后增加池化层
            if i == 1 or i == 3 or i == 7 or i == 11 or i == 15:
                layers += [nn.MaxPool2d(2, 2)]
                # 第10个卷积后保持和前边的通道数一致，都为512，其余加倍
                if i != 11:
                    out_dim *= 2
        self.features = nn.Sequential(*layers)

    # def forward(self, x):
    #     h_relu1 = self.features[:self.layer_indexs[0]](x)
    #     h_relu2 = self.features[self.layer_indexs[0]:self.layer_indexs[1]](h_relu1)
    #     h_relu3 = self.features[self.layer_indexs[1]:self.layer_indexs[2]](h_relu2)
    #     h_relu4 = self.features[self.layer_indexs[2]:self.layer_indexs[3]](h_relu3)
    #     h_relu5 = self.features[self.layer_indexs[3]:self.layer_indexs[4]](h_relu4)
    #     out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
    #     return out

    def forward(self, x):
        out = []
        for i in range(len(self.layer_indexs)):
            if i == 0:
                x = self.features[:self.layer_indexs[0]+1](x)
            else:
                x = self.features[self.layer_indexs[i-1]+1:self.layer_indexs[i]+1](x)
            out.append(x)
        # print("out:",len(out))
        return out


# 计算指定的组合模块的感知损失
class PerceptualLoss(nn.Module):
    def __init__(self,weights=[1.0 / 2, 1.0],layer_indexs=[5, 22]):
        super(PerceptualLoss, self).__init__()
        self.criterion = nn.L1Loss().to(gpu_device)#.cuda()
        self.weights = weights
        self.layer_indexs = layer_indexs
        self.vgg = VGG(self.layer_indexs)
        # print(self.vgg)
        # print(vgg19(pretrained = True))
        self.vgg.features.load_state_dict(vgg19(pretrained=True).features.state_dict())
        self.vgg.to(gpu_device)#.cuda()
        # print(self.vgg)
        self.vgg.eval()
        # print("vggg:",vgg16(pretrained = True).features[0].state_dict())
        # print("vgg:",self.vgg.features[0].state_dict())
        # 冻结参数
        for parm in self.vgg.parameters():
            parm.requires_grad = False

    def forward(self, yPred, yGT):
        yPred = yPred.to(gpu_device)#.cuda()
        yGT = yGT.to(gpu_device)#.cuda()
        yPred_vgg, yGT_vgg = self.vgg(yPred), self.vgg(yGT)
        loss = 0
        for i in range(len(yPred_vgg)):
            loss += self.weights[i] * intensity_loss(yPred_vgg[i], yGT_vgg[i],l_num=2)
        return loss

    # def forward(self, y, y_):
    #     f1 = self.vgg.forward(y.float().to(gpu_device)#.cuda())
    #     f2 = self.vgg.forward(y_.float().to(gpu_device)#.cuda())
    #     loss = self.criterion(f1, f2)
    #     # print("f1:",f1.shape)
    #     # import matplotlib.pyplot as plt
    #     # plt.subplot(131)
    #     # plt.imshow(f1[0,0, ...].cpu().detach().numpy(),cmap='gray')
    #     # plt.subplot(132)
    #     # plt.imshow(f2[0,0, ...].cpu().detach().numpy(),cmap='gray')
    #     # plt.subplot(133)
    #     # plt.imshow(y_[0, ...].permute(1, 2, 0).cpu().detach().numpy()/255.,cmap='gray')
    #     # plt.show()
    #     return loss


class Total_loss(nn.Module):
    def __init__(self, lam_appearance, lam_perception, lam_mask, lam_mesh):
        super().__init__()
        self.lam_appearance = lam_appearance
        self.lam_ssim = 5e-6
        self.lam_mask = lam_mask
        self.lam_mesh = lam_mesh
        self.lam_primary_weight = 2
        self.lam_super_weight = 1
        # weights = [0.1, 0.1, 1.0, 1.0, 1.0]
        # layer_indexs = [2, 7, 16, 25, 34]
        weights = [1.0]
        layer_indexs = [22]
        self.perceptual_loss = PerceptualLoss(weights=weights, layer_indexs=layer_indexs).to(gpu_device)#.cuda()
        #self.tv_loss = TV_Loss().to(gpu_device)#.cuda()
        self.ssim_loss = SSIM().to(gpu_device)#.cuda()
        self.grad_loss = GradLoss().to(gpu_device)#.cuda()
        self.l1_loss = nn.L1Loss().to(gpu_device)#.cuda()

    def forward(self, mesh_primary, warp_image_primary, warp_mask_primary, gt, super_image,super_gt_img):
        # print("gt:",gt.shape)
        # print("warp_image_primary:", warp_image_primary.shape)
        # import matplotlib.pyplot as plt
        # plt.subplot(121)
        # plt.imshow(gt[0,0,:,:].cpu().detach().numpy(),cmap='gray')
        # plt.subplot(122)
        # plt.imshow(warp_image_primary[0, 0,:,:].cpu().detach().numpy(),cmap='gray')
        # plt.show()

        device = gt.device
        # mesh_primary = mesh_primary.to(gpu_device)#.cuda()
        warp_image_primary = warp_image_primary.to(gpu_device)#.cuda()
        warp_mask_primary = warp_mask_primary.to(gpu_device)#.cuda()
        super_image = super_image.to(gpu_device)#.cuda()
        # define mesh
        mesh_loss = intra_grid_loss(mesh_primary) + inter_grid_loss(mesh_primary)
        # define appearance loss (loss 1 of of the content term)
        appearance_loss = self.l1_loss(warp_image_primary, gt)
        # define perception loss (loss 2 of of the content term)
        perception_loss = self.perceptual_loss(warp_image_primary*255., gt*255.)
        # ssim_loss = 1 - self.ssim_loss(warp_image_primary, gt)
        # define boundary term
        # mask_loss = intensity_loss(gen_frames=warp_mask_primary, gt_frames=torch.ones_like(warp_mask_primary), l_num=1)
        # mask_loss = self.l1_loss(warp_mask_primary, torch.ones_like(warp_mask_primary))
        # define grad
        grad_loss = self.grad_loss(warp_image_primary,gt)
        # super image loss
        super_app = self.l1_loss(super_image, super_gt_img)
        # super_perception_loss = self.perceptual_loss(super_image * 255., super_gt_img * 255.)
        super_ssim = 1 - self.ssim_loss(super_image,super_gt_img)
        super_grad = self.grad_loss(super_image,super_gt_img)
        super_img_loss = super_app * 1 + super_ssim * 0.2 + super_grad * 0.1
        primary_img_loss = appearance_loss * self.lam_appearance + perception_loss * self.lam_ssim + grad_loss * 0.1 + mesh_loss * self.lam_mesh #+ mask_loss * self.lam_mask
        # perception_loss = super_img_loss
        # total loss
        g_loss = primary_img_loss * self.lam_primary_weight + super_img_loss * self.lam_super_weight
        # g_loss = super_img_loss * self.lam_super_weight
        return g_loss*10,primary_img_loss* self.lam_primary_weight*10,super_img_loss* self.lam_super_weight *10


# pixel-level loss (l_num=1 for L1 loss, l_num=2 for L2 loss, ......)
def intensity_loss(gen_frames, gt_frames, l_num):
    return torch.mean(torch.abs((gen_frames - gt_frames) ** l_num))


# intra-grid constraint
def intra_grid_loss(pts):
    batch_size = pts.shape[0]

    delta_x = pts[:, :, 0:grid_w, 0] - pts[:, :, 1:grid_w + 1, 0]
    delta_y = pts[:, 0:grid_h, :, 1] - pts[:, 1:grid_h + 1, :, 1]

    loss_x = F.relu(delta_x + min_w)
    loss_y = F.relu(delta_y + min_h)

    loss = torch.mean(loss_x) + torch.mean(loss_y)
    return loss


# inter-grid constraint
def inter_grid_loss(train_mesh):
    w_edges = train_mesh[:, :, 0:grid_w, :] - train_mesh[:, :, 1:grid_w + 1, :]
    cos_w = torch.sum(w_edges[:, :, 0:grid_w - 1, :] * w_edges[:, :, 1:grid_w, :], 3) / \
            (torch.sqrt(torch.sum(w_edges[:, :, 0:grid_w - 1, :] * w_edges[:, :, 0:grid_w - 1, :], 3))
             * torch.sqrt(torch.sum(w_edges[:, :, 1:grid_w, :] * w_edges[:, :, 1:grid_w, :], 3)))
    # print("cos_w.shape")
    # print(cos_w.shape)
    delta_w_angle = 1 - cos_w

    h_edges = train_mesh[:, 0:grid_h, :, :] - train_mesh[:, 1:grid_h + 1, :, :]
    cos_h = torch.sum(h_edges[:, 0:grid_h - 1, :, :] * h_edges[:, 1:grid_h, :, :], 3) / \
            (torch.sqrt(torch.sum(h_edges[:, 0:grid_h - 1, :, :] * h_edges[:, 0:grid_h - 1, :, :], 3))
             * torch.sqrt(torch.sum(h_edges[:, 1:grid_h, :, :] * h_edges[:, 1:grid_h, :, :], 3)))
    delta_h_angle = 1 - cos_h

    loss = torch.mean(delta_w_angle) + torch.mean(delta_h_angle)
    return loss



class Distill_loss(nn.Module):
    def __init__(self, lam_appearance, lam_perception, lam_mask, lam_mesh):
        super().__init__()
        self.lam_appearance = lam_appearance
        self.lam_ssim = 5e-6
        self.lam_mask = lam_mask
        self.lam_mesh = lam_mesh
        self.lam_primary_weight = 2
        self.lam_super_weight = 1
        # weights = [0.1, 0.1, 1.0, 1.0, 1.0]
        # layer_indexs = [2, 7, 16, 25, 34]
        weights = [1.0]
        layer_indexs = [22]
        self.perceptual_loss = PerceptualLoss(weights=weights, layer_indexs=layer_indexs).to(gpu_device)#.cuda()
        self.grad_loss = GradLoss().to(gpu_device)#.cuda()
        self.l1_loss = nn.L1Loss().to(gpu_device)#.cuda()

    # def forward(self, mesh_primary_s, warp_image_primary_s, mesh_primary_t, warp_image_primary_t, mesh_final_s, warp_image_final_s, gt):
    #     # define mesh
    #     mesh_loss = intra_grid_loss(mesh_final_s) + inter_grid_loss(mesh_final_s)
    #     # define appearance loss (loss 1 of of the content term)
    #     appearance_loss = self.l1_loss(warp_image_final_s, gt)
    #     # define perception loss (loss 2 of of the content term)
    #     perception_loss = self.perceptual_loss(warp_image_final_s*255., gt*255.)
    #     # destill loss
    #     mesh_loss2 = self.l1_loss(mesh_primary_s, mesh_primary_t)
    #     appearance_loss2 = self.l1_loss(warp_image_primary_s, warp_image_primary_t)
    #     # perception_loss2 = self.perceptual_loss(warp_image_primary_s*255., warp_image_primary_t*255.)
    #     distill_loss = mesh_loss2 + appearance_loss2 * self.lam_appearance# + perception_loss2 * self.lam_ssim

    #     primary_img_loss = appearance_loss * self.lam_appearance + perception_loss * self.lam_ssim + mesh_loss * self.lam_mesh
    #     # perception_loss = super_img_loss
    #     # total loss
    #     g_loss = primary_img_loss * self.lam_primary_weight  + distill_loss
    #     # g_loss = super_img_loss * self.lam_super_weight
    #     return g_loss*10,primary_img_loss* self.lam_primary_weight*10, distill_loss

    # def forward(self,mesh_final_s, warp_image_final_s, super_image_t, gt):
    #     # define mesh
    #     mesh_loss = intra_grid_loss(mesh_final_s) + inter_grid_loss(mesh_final_s)
    #     # define appearance loss (loss 1 of of the content term)
    #     appearance_loss = self.l1_loss(warp_image_final_s, super_image_t)
    #     # define perception loss (loss 2 of of the content term)
    #     perception_loss = self.perceptual_loss(warp_image_final_s*255., super_image_t*255.)
    #     # perception_loss2 = self.perceptual_loss(warp_image_primary_s*255., warp_image_primary_t*255.)

    #     primary_img_loss = appearance_loss * self.lam_appearance + perception_loss * self.lam_ssim + mesh_loss * self.lam_mesh
    #     # perception_loss = super_img_loss
    #     # total loss
    #     g_loss = primary_img_loss * self.lam_primary_weight
    #     # g_loss = super_img_loss * self.lam_super_weight
    #     return g_loss*10, appearance_loss * self.lam_appearance*10, perception_loss * self.lam_ssim *10

    def forward(self,mesh_final, warp_image_final, ds_mesh, img_gt):
        # print("mesh_final:",mesh_final.shape)
        # print("ds_mesh:",ds_mesh.shape)
        # define mesh
        mesh_loss = intra_grid_loss(mesh_final) + inter_grid_loss(mesh_final)
        # define appearance loss (loss 1 of of the content term)
        appearance_loss = self.l1_loss(warp_image_final, img_gt)
        # define perception loss (loss 2 of of the content term)
        perception_loss = self.perceptual_loss(warp_image_final*255., img_gt*255.)
        # define
        primary_img_loss = appearance_loss * self.lam_appearance + perception_loss * self.lam_ssim + mesh_loss * self.lam_mesh
        # perception_loss = super_img_loss
        ds_loss = self.l1_loss(mesh_final, ds_mesh)
        # total loss
        g_loss = primary_img_loss * self.lam_primary_weight + ds_loss * 0.1 * self.lam_super_weight
        # g_loss = super_img_loss * self.lam_super_weight
        return g_loss*10, primary_img_loss * self.lam_primary_weight*10, ds_loss * 0.1 * self.lam_super_weight *10


class MultiTask_loss(nn.Module):
    def __init__(self, lam_appearance, lam_perception, lam_mask, lam_mesh):
        super().__init__()
        self.lam_appearance = lam_appearance
        self.lam_ssim = 5e-6
        self.lam_mask = lam_mask
        self.lam_mesh = lam_mesh
        self.lam_primary_weight = 1
        self.lam_super_weight = 0.5
        # weights = [0.1, 0.1, 1.0, 1.0, 1.0]
        # layer_indexs = [2, 7, 16, 25, 34]
        weights = [1.0]
        layer_indexs = [22]
        self.perceptual_loss = PerceptualLoss(weights=weights, layer_indexs=layer_indexs).to(gpu_device)#.cuda()
        self.grad_loss = GradLoss().to(gpu_device)#.cuda()
        self.l1_loss = nn.L1Loss().to(gpu_device)#.cuda()
        self.ssim_loss = SSIM().to(gpu_device)#.cuda()

    def forward(self,mesh_final, warp_image_final, img_gt,super_image, super_image_gt):
        # print("mesh_final:",mesh_final.shape)
        # print("ds_mesh:",ds_mesh.shape)
        # define mesh
        mesh_loss = intra_grid_loss(mesh_final) + inter_grid_loss(mesh_final)
        # define appearance loss (loss 1 of of the content term)
        appearance_loss = self.l1_loss(warp_image_final, img_gt)
        # define perception loss (loss 2 of of the content term)
        # perception_loss = self.perceptual_loss(warp_image_final*255., img_gt*255.)
        # define
        primary_img_loss = appearance_loss * self.lam_appearance  + mesh_loss * self.lam_mesh # + perception_loss * self.lam_ssim
        # perception_loss = super_img_loss
        super_app = self.l1_loss(super_image, super_image_gt)
        # super_perception_loss = self.perceptual_loss(super_image * 255., super_gt_img * 255.)
        super_ssim = 1 - self.ssim_loss(super_image, super_image_gt)
        super_img_loss = super_app * 1 + super_ssim * 0.2
        # total loss
        g_loss = primary_img_loss * self.lam_primary_weight + super_img_loss * self.lam_super_weight
        # g_loss = super_img_loss * self.lam_super_weight
        return g_loss*10, primary_img_loss * self.lam_primary_weight*10, super_img_loss * self.lam_super_weight *10

# class MultiTask_loss(nn.Module):
#     def __init__(self, lam_appearance, lam_perception, lam_mask, lam_mesh):
#         super().__init__()
#         self.lam_appearance = lam_appearance
#         self.lam_ssim = 5e-6
#         self.lam_mask = lam_mask
#         self.lam_mesh = lam_mesh
#         self.lam_primary_weight = 1
#         self.lam_super_weight = 0.5
#         # weights = [0.1, 0.1, 1.0, 1.0, 1.0]
#         # layer_indexs = [2, 7, 16, 25, 34]
#         weights = [1.0]
#         layer_indexs = [22]
#         self.perceptual_loss = PerceptualLoss(weights=weights, layer_indexs=layer_indexs).to(gpu_device)#.cuda()
#         self.grad_loss = GradLoss().to(gpu_device)#.cuda()
#         self.l1_loss = nn.L1Loss().to(gpu_device)#.cuda()
#         self.ssim_loss = SSIM().to(gpu_device)#.cuda()

#     def forward(self,mesh_final, warp_image_final, img_gt,super_image, super_image_gt):
#         # define mesh
#         mesh_loss = intra_grid_loss(mesh_final) + inter_grid_loss(mesh_final)
#         # define appearance loss (loss 1 of of the content term)
#         appearance_loss = self.l1_loss(warp_image_final, img_gt)
#         # define perception loss (loss 2 of of the content term)
#         perception_loss = self.perceptual_loss(warp_image_final*255., img_gt*255.)
#         # define
#         primary_img_loss = appearance_loss * self.lam_appearance + perception_loss * self.lam_ssim + mesh_loss * self.lam_mesh
#         # total loss
#         g_loss = primary_img_loss * self.lam_primary_weight
#         # g_loss = super_img_loss * self.lam_super_weight
#         return g_loss*10, primary_img_loss * self.lam_primary_weight*10, primary_img_loss * self.lam_primary_weight*10




