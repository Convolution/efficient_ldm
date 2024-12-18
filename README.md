# Discrete Variational Autoencoders for Synthetic Nighttime Visible Satellite Imagery
##### CSC2541 (Fall 2024)

[**Taming Transformers for High-Resolution Image Synthesis**]([https://compvis.github.io/taming-transformers/](https://github.com/Convolution/efficient_ldm.git))<br/>
[Mickell Als](https://github.com/mickyals),
[David Tomarov](https://github.com/Convolution)

**![generated_visible_gs-000000_e-000000_b-000013](https://github.com/user-attachments/assets/4272fd72-19d6-4452-ac11-e0b789d03429)
![IR_input_gs-000000_e-000000_b-000013](https://github.com/user-attachments/assets/50b31793-ff64-47fb-9a4a-63307eea29b2)


tl;dr** We replaced the generator of the model proposed in "Nighttime Reflectance Generation in the Visible" with a VQGAN, and created synthetic satellite images in the visible spectrum, from IR images.

## Requirements
A suitable [conda](https://conda.io/) environment named `taming` can be created
and activated with:

```
conda env create -n taming -f environment.yaml
conda activate taming
```

Please follow these steps to follow to make this work:
1. install the repo with `conda env create -n taming -f environment.yaml`, `conda activate taming` and `pip install -e .`
2. put your .npy files in a folder `your_folder`
3. create 2 text files a `train.txt` and `test.txt` that point to the files in your training and test set respectively (for example `find $(pwd)/your_folder -name "*.jpg" > train.txt`)
4. adapt `configs/custom_vqgan.yaml` to point to these 2 files
5. run `python main.py --base configs/custom_vqgan.yaml -t True --gpus 0,1` to
   train on two GPUs. Use `--gpus 0,` (with a trailing comma) to train on a single GPU.


## Dataset Construction and Description
1. build a dataset using the `ThreddsData` class from `create_datasets.py`.
2. use the functions within `goes_helper.py` to find and remove files with NaN values and finally to split your data into training and test sets.


There are 10 channels of 500x500 npy files. Mostly preprocessed and ready for training, only normalisation may be required.
Stored as 10x500x500 matrices

| Array Channel Indices | ABI Band No. | Approx. Central Wavelength (µm) | Band "Nickname"         | Band Type  |
|-----------------------|--------------|---------------------------------|-------------------------|-----------|
| 0                     | 1            | 0.47                            | "Blue" Band             | Visible   |
| 1                     | 2            | 0.64                            | "Red" Band              | Visible   |
| 2                     | 3            | 0.86                            | "Veggie" Band           | Near-IR   |
| 3                     | 8            | 6.20                            | "Upper Level Water"     | IR        |
| 4                     | 9            | 6.90                            | "Mid Level Water"       | IR        |
| 5                     | 10           | 7.30                            | "Lower Level Water"     | IR        |
| 6                     | 11           | 8.40                            | "Cloud Top Phase"       | IR        |
| 7                     | 13           | 10.30                           | "Clean" IR Longwave     | IR        |
| 8                     | 14           | 11.20                           | IR Longwave             | IR        |
| 9                     | N/A          | N/A                             | "Green"***              | Visible   |

		


### Training
1. To train your model, first adapt the `custom_vqgan.yaml` file.
2. run `python main.py --base configs/custom_vqgan.yaml -t True --gpus 0,1` to
   train on two GPUs. Use `--gpus 0,` (with a trailing comma) to train on a single GPU. Note there are other useful training arguments described within `main.py`
   Image logging by default is set to 1 but this can be changed to any number by specifying the `batch_frequency` within `main.py`

