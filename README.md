# PGRINet
Code for the paper titled "Joint pluralistic generation and realistic inpainting of occluded facial images".


# Getting started
## Installation

- Create conda environment:

```
conda create -n PGRINet python=3.7
conda activate PGRINet
pip install visdom dominate
```
- Clone this repo:

```
git clone https://github.com/sys706/PGRINet
cd PGRINet
```

## Datasets
Download [images](https://drive.google.com/file/d/11dbMGTBCt5EApq3oC0knv2FwMbSIhy1e/view?usp=drive_link) and masks [[mask_eyeglasses](https://drive.google.com/file/d/1ARFXfw6712vSx6x335bbXKgTdzh3rqHM/view?usp=drive_link), [mask_hats](https://drive.google.com/file/d/1tzJ6kuYl_pm1zB0V_-ML9PyAyFOGNP-W/view?usp=drive_link), [mask_respirators](https://drive.google.com/file/d/10PvdYGbRKUdqahOYFkogBFHE4Tg2oBl6/view?usp=drive_link)] were constructed using our proposed landmark-based facial image generation method. 

- Pip install libs:

```
pip install -r requirements.txt
```
