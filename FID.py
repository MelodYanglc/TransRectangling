import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from pytorch_fid import fid_score

# 准备真实数据分布和生成模型的图像数据
real_images_folder = r'D:\dataResource\HomographyEstimate\DIR-D\DIR-D\testing\gt'
# generated_images_folder = r'D:\dataResource\HomographyEstimate\DIR-D\DIR-D\testing\ours'
# generated_images_folder = r'C:\Users\13777\OneDrive\文档\final_rectangling'
generated_images_folder = r'C:\Users\13777\Pictures\final_rectangling\d1\d1'

# 计算FID距离值
fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder],
                                                 device="cuda",dims=2048,batch_size=1,num_workers=0)
print('FID value:', fid_value)


