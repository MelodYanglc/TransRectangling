import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import argparse
import cv2
from net.MultiTaskModel import RectanglingNetwork,reparameterize_model
import torchvision.transforms as transforms
import random

import utils.constant as constant
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)

os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
setup_seed(2023)

grid_w = constant.GRID_W
grid_h = constant.GRID_H
    
# def draw_mesh_on_warp(warp, f_local):
    
#     #f_local[3,0,0] = f_local[3,0,0] - 2
#     #f_local[4,0,0] = f_local[4,0,0] - 4
#     #f_local[5,0,0] = f_local[5,0,0] - 6
#     #f_local[6,0,0] = f_local[6,0,0] - 8
#     #f_local[6,0,1] = f_local[6,0,1] + 7
#     # print("f_local:",f_local.shape)
#     min_w = np.minimum(np.min(f_local[:,:,0]), 0).astype(np.int32)
#     max_w = np.maximum(np.max(f_local[:,:,0]), 512).astype(np.int32)
#     min_h = np.minimum(np.min(f_local[:,:,1]), 0).astype(np.int32)
#     max_h = np.maximum(np.max(f_local[:,:,1]), 384).astype(np.int32)
#     cw = max_w - min_w
#     ch = max_h - min_h
#     # print("f_local[:,:,1]:",np.max(f_local[:,:,1]))
#     print("np.maximum(np.max(f_local[:,:,1]), 384):", np.maximum(np.max(f_local[:, :, 1]), 384))
#     # print("max_h:", max_h)

#     pic = np.ones([ch+10, cw+10, 3], np.int32)*255
#     # print("pic:", pic.shape)
#     # print("warp:", warp.shape)
#     #x = warp[:,:,0]
#     #y = warp[:,:,2]
#     #warp[:,:,0] = y
#     #warp[:,:,2] = x
#     pic[0-min_h+5:0-min_h+384+5, 0-min_w+5:0-min_w+512+5, :] = warp
    
#     warp = pic
#     # print("warp:",warp.shape)
#     f_local[:,:,0] = f_local[:,:,0] - min_w+5
#     f_local[:,:,1] = f_local[:,:,1] - min_h+5
    
    
    
#     point_color = (0, 255, 0) # BGR
#     thickness = 2
#     lineType = 8
#     #cv.circle(warp, (60, 0), 60, point_color, 0)
#     #cv.circle(warp, (f_local[0,0,0], f_local[0,0,1]), 5, point_color, 0)
#     num = 1
#     for i in range(grid_h+1):
#         for j in range(grid_w+1):
#             #cv.putText(warp, str(num), (f_local[i,j,0], f_local[i,j,1]), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
#             num = num + 1
#             if j == grid_w and i == grid_h:
#                 continue
#             elif j == grid_w:
#                 cv2.line(warp, (f_local[i,j,0], f_local[i,j,1]), (f_local[i+1,j,0], f_local[i+1,j,1]), point_color, thickness, lineType)
#             elif i == grid_h:
#                 cv2.line(warp, (f_local[i,j,0], f_local[i,j,1]), (f_local[i,j+1,0], f_local[i,j+1,1]), point_color, thickness, lineType)
#             else :
#                 cv2.line(warp, (f_local[i,j,0], f_local[i,j,1]), (f_local[i+1,j,0], f_local[i+1,j,1]), point_color, thickness, lineType)
#                 cv2.line(warp, (f_local[i,j,0], f_local[i,j,1]), (f_local[i,j+1,0], f_local[i,j+1,1]), point_color, thickness, lineType)
              
#     return warp

def draw_mesh_on_warp(warp, f_local):
    
    #f_local[3,0,0] = f_local[3,0,0] - 2
    #f_local[4,0,0] = f_local[4,0,0] - 4
    #f_local[5,0,0] = f_local[5,0,0] - 6
    #f_local[6,0,0] = f_local[6,0,0] - 8
    #f_local[6,0,1] = f_local[6,0,1] + 7
    
    min_w = np.minimum(np.min(f_local[:,:,0]), 0).astype(np.int32)
    max_w = np.maximum(np.max(f_local[:,:,0]), 512).astype(np.int32)
    min_h = np.minimum(np.min(f_local[:,:,1]), 0).astype(np.int32)
    max_h = np.maximum(np.max(f_local[:,:,1]), 384).astype(np.int32)
    cw = max_w - min_w
    ch = max_h - min_h
    
    pic = np.ones([ch+10, cw+10, 3], np.int32)*255
    #x = warp[:,:,0]
    #y = warp[:,:,2]
    #warp[:,:,0] = y
    #warp[:,:,2] = x
    pic[0-min_h+5:0-min_h+384+5, 0-min_w+5:0-min_w+512+5, :] = warp
    
    warp = pic
    f_local[:,:,0] = f_local[:,:,0] - min_w+5
    f_local[:,:,1] = f_local[:,:,1] - min_h+5
    f_local = f_local.astype(int)
    
    
    point_color = (0, 255, 0) # BGR
    thickness = 2
    lineType = 8
    #cv2.circle(warp, (60, 0), 60, point_color, 0)
    #cv2.circle(warp, (f_local[0,0,0], f_local[0,0,1]), 5, point_color, 0)
    num = 1
    for i in range(grid_h+1):
        for j in range(grid_w+1):
            #cv2.putText(warp, str(num), (f_local[i,j,0], f_local[i,j,1]), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            num = num + 1
            if j == grid_w and i == grid_h:
                continue
            elif j == grid_w:
                cv2.line(warp, (f_local[i,j,0], f_local[i,j,1]), (f_local[i+1,j,0], f_local[i+1,j,1]), point_color, thickness, lineType)
            elif i == grid_h:
                cv2.line(warp, (f_local[i,j,0], f_local[i,j,1]), (f_local[i,j+1,0], f_local[i,j+1,1]), point_color, thickness, lineType)
            else :
                cv2.line(warp, (f_local[i,j,0], f_local[i,j,1]), (f_local[i+1,j,0], f_local[i+1,j,1]), point_color, thickness, lineType)
                cv2.line(warp, (f_local[i,j,0], f_local[i,j,1]), (f_local[i,j+1,0], f_local[i,j+1,1]), point_color, thickness, lineType)
              
    return warp

def generatorRandomMesh(height,width):
    mesh_final = []
    h = height / grid_h
    w = width / grid_w
    for i in range(grid_h + 1):
        for j in range(grid_w + 1):
            ww = j * w
            hh = i * h

    ori_arr = np.array(mesh_final)
    # print(ori_arr)
    # print(ori_arr.shape)
    mesh_final = torch.from_numpy(ori_arr)
    ori_pt = mesh_final.view(grid_h + 1, grid_w + 1, 2)
    print(ori_pt)
    ori_pt = ori_pt.unsqueeze(0)
    return ori_pt

def inference_func(pathInput2,pathMask2,pathGT2):
    _origin_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    resize_w, resize_h = args.img_w,args.img_h
    index_all = list(sorted([x.split('.')[0] for x in os.listdir(pathInput2)]))
    # model = RectanglingNetwork().to(args.gpu_device)
    # model.load_state_dict(torch.load(args.save_model_name))
    # model.featureExtrator.fuse()
    # model.meshRegression.fuse()
    # model.eval()
    
    # load model
    model = RectanglingNetwork(inference_mode=True)
    # model = RectanglingNetwork()
    # model.load_state_dict(torch.load(model_path))
    pretrain_model=torch.load(args.save_model_name,map_location='cpu')
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
    length = 519
    for i in range(0, length):
        idx = index_all[i]
        input_img = cv2.imread(os.path.join(pathInput2, str(idx) + '.jpg')) / 255.
        input_img = cv2.resize(input_img, (resize_w, resize_h))

        mask_img = cv2.imread(os.path.join(pathMask2, str(idx) + '.jpg')) / 255.
        mask_img = cv2.resize(mask_img, (resize_w, resize_h))

        # gt_img = cv2.imread(os.path.join(pathGT2, str(idx) + '.jpg')) / 255.
        # gt_img = cv2.resize(gt_img, (resize_w, resize_h))
        # print("input_img:",input_img.shape)
        # test_gt = _origin_transform(gt_img).unsqueeze(0).float().to(args.device_ids[0])
        test_input = _origin_transform(input_img).unsqueeze(0).float().to(args.device_ids[0])
        test_mask = _origin_transform(mask_img).unsqueeze(0).float().to(args.device_ids[0])
        # print("test_mask:",test_mask.shape)
        # print('test input = {}'.format(test_input))
        # print('test mask = {}'.format(test_mask))
        # print('test gt = {}'.format(test_gt))
        # test_mesh_primary, test_warp_image_primary, test_warp_mask_primary = model.forward(test_input, test_mask)
        test_mesh_primary, test_warp_image_primary, warp_mask_final, super_image = model.forward(test_input, test_mask)
        # print("test_mesh_primary:",test_mesh_primary)
        # print("test_warp_image_primary:", test_warp_image_primary.shape)
        # print("test_warp_mask_primary:", test_warp_mask_primary.shape)
        # mask = test_warp_mask_primary[0].permute(1,2,0).cpu().detach().numpy()
        # import matplotlib.pyplot as plt
        # plt.imshow(mask,cmap='gray')
        # plt.show()
        mesh = test_mesh_primary[0].cpu().detach().numpy()
        # mesh = generatorRandomMesh(args.img_h,args.img_w)[0].cpu().detach().numpy()
        input_image = cv2.imread(os.path.join(pathInput2, str(idx) + '.jpg'))
        input_image = cv2.resize(input_image, (resize_w, resize_h))
        print("input_image:", input_image.shape)
        print("mesh:", mesh.shape)
        input_image = draw_mesh_on_warp(input_image, mesh)

        path = args.BasePath + "final_mesh_UDIS/" + str(idx) + ".jpg"
        cv2.imwrite(path, input_image)
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/home/meiyuan/my/dataSets/DIR-D')
    parser.add_argument('--device_ids', type=list, default=[1]) 
    parser.add_argument('--img_w', type=int, default=512)
    parser.add_argument('--img_h', type=int, default=384)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_epoch', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--BasePath', type=str, default='/home/meiyuan/my/projects/DIR_D_EXP/IRFormerRectangling/')
    parser.add_argument('--save_model_name', type=str, default='/home/meiyuan/my/projects/DIR_D_EXP/IRFormerRectangling/model_finnal/IRFormer_multiTask_model_epoch150.pkl')
    parser.add_argument('--lam_perception', type=float, default=5e-6)
    parser.add_argument('--lam_mask', type=float, default=1)
    parser.add_argument('--lam_mesh', type=float, default=1)
    parser.add_argument('--lam_appearance', type=float, default=1)
    print('<==================== Loading data ===================>\n')

    args = parser.parse_args()
    print(args)
    ##############
    pathGT2 = os.path.join(args.path, 'testing/gt')
    # pathInput2 = os.path.join(args.path, 'testing/input')
    # pathMask2 = os.path.join(args.path, 'testing/mask')
    
    pathInput2 = os.path.join(args.path, 'UDIS-testing/input')
    pathMask2 = os.path.join(args.path, 'UDIS-testing/mask')
    ##########testing###############
    inference_func(pathInput2,pathMask2,pathGT2)





                






