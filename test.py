import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import cv2
import argparse
import torch
from thop import profile
from net.MultiTaskModel import RectanglingNetwork,reparameterize_model
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)

os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
setup_seed(2023)

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
    model = RectanglingNetwork(inference_mode=True)
    # model = RectanglingNetwork()
    
    # model.load_state_dict(torch.load(model_path))

    pretrain_model=torch.load(model_path,map_location='cpu')
    # 抽出现有模型中的K,V
    model_dict=model.state_dict()
    # 新建权重字典，并更新
    state_dict={k:v for k,v in pretrain_model.items() if k in model_dict.keys()}
    # 更新现有模型的权重字典
    model_dict.update(state_dict)
    # 载入更新后的权重字典
    model.load_state_dict(model_dict)

    model = model.cuda(device=args.device_ids[0]) # 模型加载到设备0
    # model.featureExtrator.fuse()
    # print(model)
    model.eval()
    model = reparameterize_model(model)

    # test_input_t = torch.randn(1,3,384,512).float().to(args.device_ids[0])
    # test_mask_t = torch.randn(1,3,384,512).float().to(args.device_ids[0])
    # flops = FlopCountAnalysis(model, (test_input_t, test_mask_t))
    # print(flop_count_table(flops, max_depth=5))

    # mesh_final, warp_image_final, warp_mask_final, super_image = model.forward(test_input_t, test_mask_t)
    # model = reparameterize_model(model)
    # mesh_final2, warp_image_final2, warp_mask_final2, super_image2 = model.forward(test_input_t, test_mask_t)
    # print("warp_image_final:",warp_image_final)
    # print("warp_image_final2:",warp_image_final2)
    

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    re_img1 = torch.rand(1, 3, args.img_h, args.img_w).float().to(args.device_ids[0])
    mask1 = torch.rand(1, 3, args.img_h, args.img_w).float().to(args.device_ids[0])
    tensor = (re_img1, mask1)
    # 分析FLOPs
    flops, params = profile(model, inputs=tensor)
    print("Number of parameter: %.2f M" % (params / 1e6))
    print("Number of GFLOPs: %.2f GFLOPs" % (flops / 1e9))
    
    psnr_list = []
    ssim_list = []
    length = 519
    for i in range(0, length):
        idx = index_all[i]

        input_img = cv2.imread(os.path.join(pathInput2, str(idx) + '.jpg'))
        mask_img = cv2.imread(os.path.join(pathMask2, str(idx) + '.jpg'))
        gt_img = cv2.imread(os.path.join(pathGT2, str(idx) + '.jpg'))
        from PIL import Image
        input_img = Image.fromarray(input_img)
        mask_img = Image.fromarray(mask_img)
        gt_img = Image.fromarray(gt_img)
        ###
        test_input = _origin_transform(input_img).unsqueeze(0).float().to(args.device_ids[0])
        test_mask = _origin_transform(mask_img).unsqueeze(0).float().to(args.device_ids[0])
        test_gt = _origin_transform2(gt_img).unsqueeze(0).float().to(args.device_ids[0])
        
        mesh_final, warp_image_final, warp_mask_final, super_image = model.forward(test_input, test_mask)

        warp_image = super_image.squeeze(0).permute(1,2,0).cpu().detach().numpy()
        test_gt = test_gt.squeeze(0).permute(1,2,0).cpu().detach().numpy()

        I1 = warp_image
        I2 = test_gt
        psnr = compare_psnr(I1, I2 , data_range=1)
        ssim = compare_ssim(I1 , I2, data_range=1, multichannel=True)

        fusion = np.zeros_like(test_gt, dtype=np.float64)
        fusion[..., 0] = warp_image[..., 0]
        fusion[..., 1] = test_gt[..., 1] * 0.5 + warp_image[..., 1] * 0.5
        fusion[..., 2] = test_gt[..., 2]
        fusion = np.clip(fusion, 0, 1)

        # path = "./fusion_demo/" + str(i + 1).zfill(5) + ".jpg"
        # # print(fusion.shape)
        # cv2.imwrite(path, fusion * 255.)

        path = "./final_rectangling/" + str(i + 1).zfill(5) + ".jpg"
        print(warp_image.shape)
        
        # path = "./final_rectangling_woFEB_woUnet/" + str(i + 1).zfill(5) + ".jpg"
        # print(warp_image.shape)
        
        # path = "./final_rectangling_woUnet/" + str(i + 1).zfill(5) + ".jpg"
        # print(warp_image.shape)
        # warp_image = cv2.resize(warp_image, (512, 384))
        cv2.imwrite(path, warp_image*255.)
        print('i = {} / {}, psnr = {:.6f}, ssim = {:.6f}'.format(i + 1, length, psnr,ssim))

        psnr_list.append(psnr)
        ssim_list.append(ssim)

    print("===================Results Analysis==================")
    print('average psnr:', np.mean(psnr_list))
    print('average ssim:', np.mean(ssim_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=r'D:\dataResource\imageStitiching\DIR-D\DIR-D')
    parser.add_argument('--device_ids', type=list, default=[0])
    parser.add_argument('--img_h', type=int, default=384)
    parser.add_argument('--img_w', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--save_model_name', type=str, default='./model/multiTask_model_epoch150.pkl')
    parser.add_argument('--lam_perception', type=float, default=5e-6)
    parser.add_argument('--lam_mask', type=float, default=1)
    parser.add_argument('--lam_mesh', type=float, default=1)
    parser.add_argument('--lam_appearance', type=float, default=1)
    print('<==================== Loading data ===================>\n')

    args = parser.parse_args()
    print(args)
    ##############
    pathGT2 = os.path.join(args.path, 'testing/gt')
    pathInput2 = os.path.join(args.path, 'testing/input')
    pathMask2 = os.path.join(args.path, 'testing/mask')
    model_path = args.save_model_name
    # test
    inference_func(pathInput2,pathMask2,pathGT2,model_path)

        
        
        


    






