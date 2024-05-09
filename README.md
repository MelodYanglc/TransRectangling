# <p align="center">Image Rectangling Network Based on Reparameterized Transformer and Assisted Learning</p>
<p align="center">Lichun Yang', Bin Tian*, Tianyin Zhang', Jiu Yong*, Jianwu Dang*</p>
<p align="center">* the School of Electronic and Information Engineering, Lanzhou Jiaotong University</p>
<p align="center">' the Key Lab of Opt-Electronic Technology and Intelligent Control of Ministry of Education, Lanzhou Jiaotong University</p>

![image](./main/1.png)

## Datasets(DIR-D)
We use the DIS-D dataset to train and evaluate our method. Please refer to [DIR-D](https://github.com/nie-lang/DeepRectangling) for more details about this dataset


## Code
#### Requirement
* numpy 1.19.5
* pytorch 1.7.1
* scikit-image 0.15.0
* tensorboard 2.9.0

We implement this work with Ubuntu, 2080Ti, and CUDA11. Refer to [environment.yml]() for more details.
## Train
Step 1: Train the network
Modify the 'utils/constant.py' to set the 'GRID_W'"GRID_H"'GPU'. In our experiment, we set"GRID_W' to 8 and 'GRID_H' to 6.
Step 2: Change the path to your own file, then run the train.py file

## Test
Step 1: Change the path to your own file, then run the test.py file

## Meta
If you have any questions about this project, please feel free to drop me an email.

Lichun Yang -- ylc1377759045@163.com
