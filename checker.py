import os
import numpy as np


def check_npy_files_for_nan(folder_path, output_txt_path):
    nan_file_count = 0
    not_nan_file_count = 0
    nan_file_names = []

    # Loop through files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npy'):
            file_path = os.path.join(folder_path, file_name)

            # Load .npy file
            try:
                data = np.load(file_path)

                # Check for NaN values
                if np.isnan(data).any():
                    nan_file_count += 1
                    nan_file_names.append(file_name)
                else:
                    not_nan_file_count += 1
            except Exception as e:
                print(f"Could not load {file_name}: {e}")

    # Write names of files with NaNs to a text file
    with open(output_txt_path, 'w') as f:
        for name in nan_file_names:
            f.write(name + '\n')

    # Print counts of files
    print(f"Files with NaN values: {nan_file_count}")
    print(f"Files without NaN values: {not_nan_file_count}")


def delete_files_from_list(folder_path, txt_file_path):
    with open(txt_file_path, 'r') as file:
        file_names = file.readlines()

    deleted_count = 0
    not_found_count = 0

    for file_name in file_names:
        # Strip whitespace and join path
        file_name = file_name.strip()
        file_path = os.path.join(folder_path, file_name)

        # Check if file exists, then delete
        if os.path.exists(file_path):
            os.remove(file_path)
            deleted_count += 1
            print(f"Deleted: {file_path}")
        else:
            not_found_count += 1
            print(f"File not found: {file_path}")

    # Summary
    print(f"\nTotal files deleted: {deleted_count}")
    print(f"Total files not found: {not_found_count}")


def calculate_mean_std(directory, output_file):
    # Initialize arrays to store sums and counts for mean and std calculation
    channel_sums = np.zeros(17)
    channel_sums_squared = np.zeros(17)
    total_count = 0

    # Loop through all .npy files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            # Load the .npy file
            data = np.load(os.path.join(directory, filename))

            # Ensure the data has the expected shape (C, H, W) for 17 channels
            if data.shape[0] != 17:
                print(f"Warning: {filename} does not have 17 channels, skipping.")
                continue

            # Calculate mean and std for this file
            channel_sums += data.sum(axis=(1, 2))  # Sum across height and width
            channel_sums_squared += (data ** 2).sum(axis=(1, 2))
            total_count += data.shape[1] * data.shape[2]  # H * W for each channel

    # Calculate mean for each channel
    means = channel_sums / total_count

    # Calculate std for each channel
    stds = np.sqrt((channel_sums_squared / total_count) - (means ** 2))

    # Prepare output
    with open(output_file, 'w') as f:
        f.write("Channel\tMean\tStd\n")
        for i in range(17):
            f.write(f"{i + 1}\t{means[i]:.6f}\t{stds[i]:.6f}\n")

    print(f"Mean and standard deviation calculated for {total_count} pixels across all channels.")
    print(f"Results saved to {output_file}")


def normalise_data(folder_path):
    max_min = [
        [187, 260],  # Band 8
        [181, 270],  # Band 9
        [171, 277],  # Band 10
        [181, 323],  # Band 11
        [181, 330],  # Band 13
        [172, 330]  # Band 14
    ]

    bands = [7, 8, 9, 10, 12, 13]  # Adjusted to reflect 0-based indexing

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npy'):
            file_path = os.path.join(folder_path, file_name)

            try:
                # Load the .npy file
                data = np.load(file_path)

                # Ensure data shape is (17, 500, 500)
                if data.shape != (17, 500, 500):
                    print(f"Skipping {file_name}: Unexpected shape {data.shape}.")
                    continue

                # Normalize the specified bands
                for j, i in enumerate(bands):
                    # Apply min-max normalization
                    data[i] = (data[i] - max_min[j][0]) / (max_min[j][1] - max_min[j][0])
                    # Clip values to keep them within [0, 1]
                    data[i] = np.clip(data[i], 0, 1)

                # Save the normalized data back to the .npy file
                if data.shape != (17, 500, 500):
                    print('error, stopping before saving')
                    break
                else:
                    np.save(file_path, data)
                print(f"Normalized {file_name}.")

            except Exception as e:
                print(f"Could not process {file_name}: {e}")


# Example usage

if __name__ == '__main__':
    folder_path = '../shared_data/temp_dataset/goes_images'
    output_txt_file = 'delete_list.txt'

    # Call the function
    # check_npy_files_for_nan(folder_path, output_txt_file)
    delete_files_from_list(folder_path, output_txt_file)