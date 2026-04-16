# PGRINet
Code for the paper titled "Joint pluralistic generation and realistic inpainting of occluded facial images".


## Datasets
- Download [images](https://drive.google.com/file/d/11dbMGTBCt5EApq3oC0knv2FwMbSIhy1e/view?usp=drive_link) and masks [[mask_eyeglasses](https://drive.google.com/file/d/1ARFXfw6712vSx6x335bbXKgTdzh3rqHM/view?usp=drive_link), [mask_hats](https://drive.google.com/file/d/1tzJ6kuYl_pm1zB0V_-ML9PyAyFOGNP-W/view?usp=drive_link), [mask_respirators](https://drive.google.com/file/d/10PvdYGbRKUdqahOYFkogBFHE4Tg2oBl6/view?usp=drive_link)] were constructed using our proposed landmark-based facial image generation method.
- Irregular masks can be automatically generated during training through `/util/task.py`.


## Training
- The training consists of two steps.
- Step 1 for training the coarse inpainting model and pluralistic generation model.
- Step 2 for training the fine inpainting model.
