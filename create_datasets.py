# imports
import os
import numpy as np
import pandas as pd
from siphon.catalog import TDSCatalog
import xarray as xr
import warnings
import torch
# from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import albumentations
import cv2

warnings.filterwarnings("ignore")


class ThreddsData():
    def __init__(self, timeframe, catalog_url, storage_path=None,
                 openDAP_url_regex='https://redoak.cs.toronto.edu/twitcher/ows/proxy/thredds/dodsC/datasets/GOES17/ABI-L2-MCMIPM/'):
        self.years = timeframe[0]
        self.days_of_year = timeframe[1]
        self.hour_of_day = timeframe[2]
        self.catalog = catalog_url
        self.openDAP = openDAP_url_regex

        # Create the storage path
        # Set storage_path to the current working directory if not provided
        self.storage_path = storage_path if storage_path is not None else os.path.join(os.getcwd(), 'Data_Train')
        os.makedirs(self.storage_path, exist_ok=True)

    def get_data(self):

        base_url = 'https://redoak.cs.toronto.edu/twitcher/ows/proxy/thredds/catalog/datasets/GOES17/ABI-L2-MCMIPM/'
        images_path = os.path.join(self.storage_path, 'goes_images')
        metadata_path = os.path.join(self.storage_path, 'metadata')
        OpenDAP_urls = []
        npy_total = 0
        meta_total = 0

        for year in self.years:
            for day in self.days_of_year:
                for hour in self.hour_of_day:
                    day_str = str(day).zfill(3)
                    url = f'{base_url}{year}/{day_str}/{hour}/catalog.html'

                    try:
                        catalog = TDSCatalog(url)
                        datasets = [file for file in catalog.datasets if 'MCMIPM1' in str(file)]
                        datasets = sorted(datasets)
                        print(f'Found {len(datasets)} datasets for {year} on day {day_str} at hour {hour}')
                    except Exception as e:
                        print(f'Error loading catalog {e}')
                        continue

                    try:
                        os.makedirs(images_path, exist_ok=True)
                        os.makedirs(metadata_path, exist_ok=True)

                        # get images
                        for i in range(0, len(datasets), 5):
                            url = f'{self.openDAP}{year}/{day_str}/{hour}/{datasets[i]}'
                            x_dataset = xr.open_dataset(url)

                            npy = self.process_xarray(x_dataset)

                            name = f'MCMIPM1_{year}_{day_str}_{hour}_{datasets[i][-8:-3]}'
                            np.save(os.path.join(images_path, f'{name}.npy'), npy)

                            npy_total += 1

                            # get metadata
                            meta = x_dataset['CMI_C01']

                            meta.to_netcdf(os.path.join(metadata_path, f'{name}'))
                            print(f'Successfully downloaded dataset {datasets[i][:-3]}')

                            meta_total += 1

                            OpenDAP_urls.append(url)

                    except Exception as e:
                        print(f'Error loading dataset: {e}')
                        continue
        df = {'openDAP URL': OpenDAP_urls}
        df = pd.DataFrame(df)
        df.to_csv(os.path.join(self.storage_path, 'openDAP_urls.csv'), index=True)

        print(f'Done! A total of {npy_total} were created and a total of {meta_total} were saved.')

    def process_xarray(self, xarray_dataset, variable_list=None):
        if variable_list is None:
            variable_list = ['CMI_C01', 'CMI_C02', 'CMI_C03', 'CMI_C08', 'CMI_C09', 'CMI_C10',
                             'CMI_C11', 'CMI_C13', 'CMI_C14']

        try:
            selected_data = xarray_dataset[variable_list]

            for i in variable_list[:3]:
                selected_data[i] = selected_data[i].clip(0, 1)

            for i in variable_list[3:]:
                min_value = xarray_dataset[f"min_brightness_temperature_{i[4:]}"].attrs['valid_range'][0]
                max_value = xarray_dataset[f"min_brightness_temperature_{i[4:]}"].attrs['valid_range'][1]

                normalised = (selected_data[i] - min_value) / (max_value - min_value)

                selected_data[i] = normalised

            numpy_data_list = [selected_data[variable] for variable in variable_list]

            green_band = 0.48358168 * xarray_dataset['CMI_C02'] + 0.45706946 * xarray_dataset['CMI_C01'] + 0.06038137 * \
                         xarray_dataset['CMI_C03']

            numpy_data_list.append(green_band)

            variable_array = np.stack(numpy_data_list, axis=0)

            assert variable_array.shape == (10, 500, 500)

            return variable_array
        except Exception as e:
            print(f'Error processing data: {e}')
            return None


class GoesNumpyDataset(Dataset):

    def __init__(self, data_dir, size=None, downscale_f=4, x_channels=None, y_channels=None):
        self.base = self.get_base(data_dir)
        assert size
        assert (size / downscale_f).is_integer()
        self.size = size
        self.rescaler = albumentations.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_CUBIC)

        self.x_idxs = x_channels
        self.y_idxs = y_channels

    def get_base(self, data_dir):
        # Get all numpy file paths from the data directory
        file_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.npy')]
        return file_paths

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.base)

    def split_ids(self, split=0.1, seed=42):
        train_idx, val_idx = train_test_split(np.arange(len(self.base)), test_size=split, shuffle=True,
                                              random_state=seed)
        return train_idx, val_idx

    def __getitem__(self, i):
        # load data
        example = self.base[i]
        data = np.load(example).astype(np.float32)

        # make inut and output arrays
        input = data[self.x_idxs]
        output = data[self.y_idxs]

        # rescale C H W (0, 1, 2) -> H W C (1, 2, 0) -> C H W (2, 0, 1)
        input = self.rescaler(image=input.transpose(1, 2, 0))["image"].transpose(2, 0, 1)
        output = self.rescaler(image=output.transpose(1, 2, 0))["image"].transpose(2, 0, 1)

        # clip values to [0, 1]
        input = np.clip(input, 0, 1)
        output = np.clip(output, 0, 1)

        # normalize to [-1, 1]
        res = {}
        #res["input"] = (2 * input - 1).astype(np.float32)
        #res["output"] = (2 * output - 1).astype(np.float32)
        res["input"] = input.astype(np.float32)
        res["output"] = output.astype(np.float32)

        return res


class GOESTrainingDataset(GoesNumpyDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self, data_dir):
        train_idx, _ = self.split_ids()
        file_paths = super().get_base(data_dir)
        return Subset(file_paths, train_idx)


class GOESValidationDataset(GoesNumpyDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self, data_dir):
        _, val_idx = self.split_ids()
        file_paths = super().get_base(data_dir)
        return Subset(file_paths, val_idx)

def dataset_to_jpeg(ds, save_path):
    for i in range(len(ds)):
        cv2.imwrite(f"{save_path}_output_{i}.jpg", ds[i]['output'].reshape(512,512,3))

def create_split_files(path):
    l = os.listdir(path)
    i = 0
    split = int(len(l) * 0.8)

    # Write names of files with NaNs to a text file
    with open('xx_train.txt', 'w') as f:
        while i <= split:
            f.write(path + '/' + l[i] + '\n')
            i += 1
            print(i)

    with open('xx_test.txt', 'w') as f:
        while i < len(l):
            f.write(path + '/' + l[i] + '\n')
            i += 1
            print(i)

if __name__ == "__main__":
    path = '../shared_data/temp_dataset/goes_images'
    ds = GoesNumpyDataset(path, size=512, x_channels=[3,4,5,6,7,8], y_channels=[1,9,0])

    save_path = '../shared_data/temp_dataset/jpeg'
    # dataset_to_jpeg(ds, save_path)


    create_split_files(save_path)
    print('done')
    # years = [2019]
    # days = np.arange(1,25, 5)
    # hours = ['18','19','20','21']
    # catalog_url = 'https://redoak.cs.toronto.edu/twitcher/ows/proxy/thredds/catalog/datasets/GOES17/ABI-L2-MCMIPM/catalog.xml'
    # timeframe = [years, days, hours]
    # thredds = ThreddsData(timeframe, catalog_url, storage_path='../shared_data/temp_dataset')
    # thredds.get_data()