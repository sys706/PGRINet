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

- Pip install libs:

```
pip install -r requirements.txt
```

## Datasets
- Download [images](https://drive.google.com/file/d/11dbMGTBCt5EApq3oC0knv2FwMbSIhy1e/view?usp=drive_link) and masks [[mask_eyeglasses](https://drive.google.com/file/d/1ARFXfw6712vSx6x335bbXKgTdzh3rqHM/view?usp=drive_link), [mask_hats](https://drive.google.com/file/d/1tzJ6kuYl_pm1zB0V_-ML9PyAyFOGNP-W/view?usp=drive_link), [mask_respirators](https://drive.google.com/file/d/10PvdYGbRKUdqahOYFkogBFHE4Tg2oBl6/view?usp=drive_link)] were constructed using our proposed landmark-based facial image generation method.
- Irregular masks can be automatically generated during training in `/util/task.py`.


## Training
- The training consists of two steps.
- Step 1 for training the coarse inpainting model and pluralistic generation model.
- Step 2 for training the fine inpainting model.

### Step 1:

```
python train_coarse.py --model 'TCAM_woFine' --niter 20 --batchSize 4 --checkpoints_dir [save_coarse_model_path] --name [exp_coarse_name] --mask_file [mask_path] --img_file [image_path] --trainData [training_image_names]
```


### Step 2:

```
python train_fine.py --model 'TCAM_woFine' --model_fine 'TCAM_Fine' --niter 20 --batchSize 2 --checkpoints_dir [saved_coarse_model_path] --name [exp_coarse_name] --checkpoints_dir_fine [save_fine_model_path] --name_fine [exp_fine_name] --mask_file [mask_path] --img_file [image_path] --trainData [training_image_names]
```


## Testing
- The testing consists of two steps.
- Step 1 for reconstructing one coarse result from the trained coarse inpainting model and generating diverse results from the trained pluralistic generation model.
- Step 2 for reconstructing a high-fidelity result from the trained fine inpainting model.

### Step 1:

```
python test_coarse.py --model 'TCAM_Fine' --batchSize 1 --how_many [number_of_diverse_results_to_generation] --nsampling [number_of_diverse_results_to_sample] --which_iter 'latest' --results_dir [saved_result_path] --checkpoints_dir [save_coarse_model_path] --name [exp_coarse_name] --mask_file [mask_path] --img_file [image_path] --trainData [testing_image_names]
```
Notes of testing: 
- `--how_many`: How many completion results do you want?


### Step 2:

```
python test_fine.py --model 'TCAM_woFine' --model_fine 'TCAM_Fine' --batchSize 1 --which_iter 'latest' --results_dir [saved_result_path] --checkpoints_dir [saved_coarse_model_path] --name [exp_coarse_name] --checkpoints_dir_fine [saved_fine_model_path] --name_fine [exp_fine_name] --mask_file G:/sys/datasets/image-inpainting/mask_eyeglasses --mask_file [mask_path] --img_file [image_path] --trainData [testing_image_names]
```
