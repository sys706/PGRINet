# PGRINet
Code for the paper titled "Joint pluralistic generation and realistic inpainting of occluded facial images".


# Getting started
## Installation
This code was tested with Pytoch 1.8.1 CUDA 11.1, Python 3.6 and Ubuntu 18.04

- Create conda environment:

```
conda create -n inpainting-py36 python=3.6
conda deactivate
conda activate inpainting-py36
pip install visdom dominate
```
- Clone this repo:

```
git clone https://github.com/huangwenwenlili/spa-former
cd spa-former
```

- Pip install libs:

```
pip install -r requirements.txt
```

## Datasets
- ```Paris StreetView```: It contains buildings of Paris of natural digital images. 14900 training images and 100 testing images. [Paris](https://github.com/pathak22/context-encoder)
- ```CelebA-HQ```: It contains celebrity face images. 30000 images. [CelebA-HQ](https://github.com/switchablenorms/CelebAMask-HQ)
- ```Places365-Standard```: It is the major part of the places2 dataset and was released by MIT. It has over 1.8 million training images and about 32K test images from 365 scene categories. We selected 1,000 pictures from the test set randomly to test the model. 

## Train
- Train the model. Input images and masks resolution are 256*256. We produce random irregular mask to corrupt images for training stage.
```
python train.py --name paris --checkpoints_dir ./checkpoints/checkpoint_paris --img_file /home/hwl/hwl/datasets/paris/paris_train_original/ --niter 261000 --batchSize 4 --lr 1e-4 --gpu_ids 0 --no_augment --no_flip --no_rotation 
```
- Set ```--mask_type``` in options/base_options.py to test various masks. ```--mask_file``` path is needed for **2 and 4 . random irregular mask**.
- ```--lr``` is learn rate, train scratch is 1e-4, finetune is 1e-5.

## Testing

- Test the model. Input images and masks resolution are 256*256. In the testing, we use [irregular mask dataset](https://github.com/NVIDIA/partialconv) to evaluate different ratios of corrupted region images.

```
python test.py  --name paris --checkpoints_dir ./checkpoints/checkpoint_paris --gpu_ids 0 --img_file your_image_path --mask_file your_mask_path --batchSize 1 --results_dir your_image_result_path
```
- Set ```--mask_type``` in options/base_options.py to test various masks. ```--mask_file``` path is needed for **3. external irregular mask**,
- The default results will be saved under the *results* folder. Set ```--results_dir``` for a new path to save the result.
- checkpooints [Baidu disk](https://pan.baidu.com/s/1Ace1zD_lUg-_KW7v_aVAOg?pwd=neg1); [Google Drive](https://drive.google.com/drive/folders/1vg6RoavdoZI8KeCeFVRVgcSg5Va4rB2P?usp=sharing)


