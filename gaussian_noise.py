import numpy as np

from moving_mnist_multiple import one_time_example


def add_noise(data, noise_ratio):
    """
    Adds Gaussian noise to each frame in the dataset, using the formula x_noise = x_origin + epsilon * std(x_origin) * sigma_R.

    Parameters:
        data (np.ndarray): The input dataset with shape (A, B, 64, 64).
        noise_ratio (float): The noise ratio to scale the standard deviation of the Gaussian noise.

    Returns:
        np.ndarray: The dataset with Gaussian noise added, with the same shape as the input (A, B, 64, 64) and pixel values as integers.
    """
    # Calculate the standard deviation of the original data
    std_dev = np.std(data)

    # Calculate the noise standard deviation (sigma_R)
    noise_std = noise_ratio * std_dev

    # Generate Gaussian noise
    noise = np.random.normal(0, noise_std, size=data.shape)

    # Apply the Gaussian noise formula to the data
    noisy_data = data + noise

    # Clip the values to ensure they are valid and convert to integer
    noisy_data = np.clip(noisy_data, 0, 255).astype(np.uint8)

    return noisy_data


def generate_noise_dataset_from_existing_dataset(dataset_path, dataset_name, noise_ratio=0.01, seed=42):
    np.random.seed(seed)
    d = np.load(dataset_path)
    # print(f"d.max: {np.max(d)}, d.min: {np.min(d)}, ")
    noise_d = add_noise(d, noise_ratio)
    # print(f"noise_d.max: {np.max(noise_d)}, noise_d.min: {np.min(noise_d)}, ")
    print(f"noise ratio = {noise_ratio}: MSE = {np.mean((noise_d - d) ** 2):.6f}")
    one_time_example(np.transpose(noise_d, axes=(1, 0, 2, 3)), f"dataset/{dataset_name}_{noise_ratio}", range(20))
    np.save(f"output_dataset/{dataset_name}_{noise_ratio}.npy", noise_d)
    print(f"saved to output_dataset/{dataset_name}_{noise_ratio}.npy")


if __name__ == "__main__":
    # generate_noise_dataset_from_existing_dataset("output_dataset/moving_mnist_binary_1_digit_1000.npy", "moving_mnist_binary_1_digit_1000", 0.01)
    # generate_noise_dataset_from_existing_dataset("output_dataset/moving_mnist_binary_1_digit_1000.npy", "moving_mnist_binary_1_digit_1000", 0.1)
    # generate_noise_dataset_from_existing_dataset("output_dataset/moving_mnist_binary_1_digit_1000.npy", "moving_mnist_binary_1_digit_1000", 0.2)
    # generate_noise_dataset_from_existing_dataset("output_dataset/moving_mnist_binary_1_digit_1000.npy", "moving_mnist_binary_1_digit_1000", 0.3)
    for noise_ration in [0.025, 0.05, 0.1, 0.2, 0.4]:
        generate_noise_dataset_from_existing_dataset("output_dataset/moving_mnist_binary_1_digit_1000.npy", "moving_mnist_binary_1_digit_1000", noise_ration)


