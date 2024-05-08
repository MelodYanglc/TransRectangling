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
import time

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

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    re_img1 = torch.rand(1, 3, args.img_h, args.img_w).float().to(args.device_ids[0])
    mask1 = torch.rand(1, 3, args.img_h, args.img_w).float().to(args.device_ids[0])
    tensor = (re_img1, mask1)
    # 分析FLOPs
    flops, params = profile(model, inputs=tensor)
    print("Number of parameter: %.2f M" % (params / 1e6))
    print("Number of GFLOPs: %.2f GFLOPs" % (flops / 1e9))
    timeTotal = 0
    numberTotal = 0
    for kk in range(11):
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
            time1 = time.time()
            mesh_final, warp_image_final, warp_mask_final, super_image = model.forward(test_input, test_mask)
            time2 = time.time()
            if kk >=1:
                timeTotal += time2 - time1
                numberTotal += 1
            print('index:', str(i+1))
        
    print("===================Results Analysis==================")
    print('average time:', timeTotal/numberTotal)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/home/meiyuan/my/dataSets/DIR-D')
    parser.add_argument('--device_ids', type=list, default=[1]) 
    parser.add_argument('--img_h', type=int, default=384)
    parser.add_argument('--img_w', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--BasePath', type=str, default='/home/meiyuan/my/projects/DIR_D_EXP/IRFormerRectangling/')
    parser.add_argument('--save_model_name', type=str, default='/home/meiyuan/my/projects/DIR_D_EXP/IRFormerRectangling/model/IRFormer_multiTask_model_epoch150.pkl')
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

        
        
        


    






