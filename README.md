# Discrete Variational Autoencoders for Synthetic Nighttime Visible Satellite Imagery
##### CSC2541 (Fall 2024)

[**Taming Transformers for High-Resolution Image Synthesis**]([https://compvis.github.io/taming-transformers/](https://github.com/Convolution/efficient_ldm.git))<br/>
[Mickell Als](https://github.com/mickyals),
[David Tomarov](https://github.com/Convolution)

**tl;dr** We replaced the generator of the model proposed in "Nighttime Reflectance Generation in the Visible" with a VQGAN, and created synthetic satellite images in the visible spectrum, from IR images.

## Requirements
A suitable [conda](https://conda.io/) environment named `taming` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate taming
```

Please follow these steps to follow to make this work:
1. install the repo with `conda env create -f environment.yaml`, `conda activate taming` and `pip install -e .`
2. put your .jpg files in a folder `your_folder`
3. create 2 text files a `xx_train.txt` and `xx_test.txt` that point to the files in your training and test set respectively (for example `find $(pwd)/your_folder -name "*.jpg" > train.txt`)
4. adapt `configs/custom_vqgan.yaml` to point to these 2 files
5. run `python main.py --base configs/custom_vqgan.yaml -t True --gpus 0,1` to
   train on two GPUs. Use `--gpus 0,` (with a trailing comma) to train on a single GPU.


## Datset Description
There are 10 channels of 500x500 npy files. Mostly preprocessed and ready for training, only normalisation may be required.
Stored as 10x500x500 matrices

 Array Channel Indices | ABI Band No. |	Approx. Central Wavelength (µm) |	Band "Nickname"  | Band Type	 |
=============================================================================================================|
		0 			      | 	1		      |		0.47 					          |	 "Blue" Band 	     |     Visible	 |
		1              |    2         |     0.64                        |    "Red" Band       |     Visible    |
		2              |    3         |     0.86                        |    "Veggie" Band    |     Near-IR    |
		3              |    8         |     6.20                        |"Upper Level Water"  |     IR         |
		4              |    9         |     6.90                        |"Mid Level Water"    |     IR         |
		5              |    10        |     7.30                        |"Lower Level Water"  |     IR         |
		6              |    11        |     8.40                        |"Cloud Top Phase"    |     IR         |
		7              |    13        |    10.30                        |"Clean" IR Longwave  |     IR         |
		8              |    14        |    11.20                        | IR Longwave         |     IR         |
		9              |    N/A       |     N/A                         | "Green"***          |     Visible    |
		


### Training
First, download the training dataset from []. To train the model, simply run `bash training_script.sh`. Follow the instructions for evaluating a pretrained model. The only difference here is that in the config file, specify a positive `base_learning_rate`, `max_epochs`, and the path to the training and testing data text files in `training_images_list_file` and `test_images_list_file` accordingly. 

## Data Preparation

## More Resources
