# <p align="center">Image Rectangling Network Based on Reparameterized Transformer and Assisted Learning</p>
<p align="center">Lichun Yang', Bin Tian*, Tianyin Zhang', Jiu Yong*, Jianwu Dang*</p>
<p align="center">* the School of Electronic and Information Engineering, Lanzhou Jiaotong University</p>
<p align="center">' the Key Lab of Opt-Electronic Technology and Intelligent Control of Ministry of Education, Lanzhou Jiaotong University</p>

## Datasets(DIR-D)
We use the DIR-D dataset to train and evaluate our method. Please refer to [DIR-D](https://github.com/nie-lang/DeepRectangling) for more details about this dataset


## Code
#### Requirement
* numpy 1.19.5
* pytorch 1.7.1
* scikit-image 0.15.0
* tensorboard 2.9.0

## Training
Modify the 'utils/constant.py' to set the 'GRID_W'"GRID_H"'GPU'. In our experiment, we set"GRID_W' to 8 and 'GRID_H' to 6. Addionally, modify the data loading path inside the train.py file to your own dataset storage path. Then run the train.py file to train the model.

## Testing
Modify the data loading path inside the test.py file to your own dataset storage path. Then run the test.py file to test the model.

Besides, if you want to see the results after the reconstruction of the assisted learning network, comment out line 41 of the code in the test.py file and uncomment line 42.

It should be added that we did not use the output of the assisted network because we thought that the results after the Assisted Learning Network generated information that was not present in the original image, which was contrary to our original intention of rectangling the original image based only on its content information. (Although the SSIM and PSNR metrics of the reconstructed image after this network would be higher.)

## Meta
If you have any questions about this project, please feel free to drop me an email.

Lichun Yang -- ylc1377759045@163.com
