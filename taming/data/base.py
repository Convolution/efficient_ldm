import bisect
import numpy as np
import albumentations
import cv2
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset


class ConcatDatasetWithIndex(ConcatDataset):
    """Modified from original pytorch code to return dataset idx"""
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx


class ImagePaths(Dataset):
    def __init__(self, paths, size=None, random_crop=False, labels=None):
        self.size = size
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class GoesNumpyDataset(Dataset):

    def __init__(self, paths, size=None, x_channels=[3,4,5,6,7,8], y_channels=[1,9,0]):

        self.paths = paths
        self.size = size
        self._length = len(paths)
        self.rescaler = albumentations.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_CUBIC)

        self.x_idxs = x_channels
        self.y_idxs = y_channels

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        data = np.load(image_path).astype(np.float32)

        # make input and output arrays
        input = data[self.x_idxs]
        output = data[self.y_idxs]

        rescale_input = self.rescaler(image=input.transpose(1, 2, 0))["image"]
        rescale_output = self.rescaler(image=output.transpose(1, 2, 0))["image"]

        input = rescale_input
        output = rescale_output

        # clip values to [0, 1]
        input = np.clip(input, 0, 1)
        output = np.clip(output, 0, 1)

        input = (2 * input - 1).astype(np.float32)
        output = (2 * output - 1).astype(np.float32)

        return input, output

    def __getitem__(self, i):
        # load data
        data_path = self.paths[i]
        input, output = self.preprocess_image(data_path)

        return {"input": input, "target": output}



class NumpyPaths(ImagePaths):
    def preprocess_image(self, image_path):
        image = np.load(image_path).squeeze(0)  # 3 x 1024 x 1024
        image = np.transpose(image, (1,2,0))
        image = Image.fromarray(image, mode="RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image
