import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import tqdm

from torchvision import datasets, transforms


def generate_moving_mnist_multiple(mnist, shape=(64, 64), seq_len=20, num_samples=1000, n_digit=2):
    digit_size = 28  # Size of a single MNIST digit

    # Container for the dataset
    dataset = torch.zeros((num_samples, seq_len, 1, *shape))

    for n in tqdm.tqdm(range(num_samples)):
        # Select digits to be consistent throughout the sequence
        digits_idx = np.random.choice(len(mnist), n_digit, replace=False)
        # digits = [torch.tensor(mnist[idx], dtype=torch.float32) for idx in digits_idx]
        digits = [mnist[idx].clone().detach() for idx in digits_idx]

        # Initialize positions and velocities
        positions = np.random.randint(0, [shape[0] - digit_size, shape[1] - digit_size], size=(n_digit, 2))
        velocities = np.random.randint(-4, 5, size=(n_digit, 2))

        for t in range(seq_len):
            frame = torch.zeros((1, *shape))

            for d in range(n_digit):
                # Update digit positions
                positions[d] += velocities[d]

                # Boundary collision detection and velocity reversal
                if positions[d, 0] <= 0 or positions[d, 0] >= shape[0] - digit_size:
                    velocities[d, 0] *= -1
                if positions[d, 1] <= 0 or positions[d, 1] >= shape[1] - digit_size:
                    velocities[d, 1] *= -1

                # Ensure positions are within bounds
                positions[d, 0] = max(0, min(positions[d, 0], shape[0] - digit_size))
                positions[d, 1] = max(0, min(positions[d, 1], shape[1] - digit_size))

                # Place the digit in the frame and handle overlaps (white pixels prevail)
                x, y = positions[d]
                digit_tensor = digits[d].unsqueeze(0)
                frame[0, x:x + digit_size, y:y + digit_size] = torch.max(frame[0, x:x + digit_size, y:y + digit_size],
                                                                         digit_tensor)

            dataset[n, t] = frame

    return dataset


def combine_images_with_edges(x, save_path, n_w=10):
    # print(x.shape)
    n_w = min(n_w, x.shape[0])
    fig, axes = plt.subplots(1, n_w, figsize=(n_w * 2, 1 * 2))
    for j in range(n_w):
        axes[j].imshow(x[j], cmap='gray')
        axes[j].axis('off')
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def one_time_example(data, save_folder, idx_list):
    # print(f"data shape: {data.shape}")

    for idx in idx_list:
        example_path = os.path.join(save_folder, str(idx))
        if not os.path.exists(example_path):
            os.makedirs(example_path)
        example_data = data[idx]
        combine_images_with_edges(example_data, f'{example_path}/all.png', 10)
        # print(example_data.shape)
        # print(f"example_data[i]: {example_data[0].shape}")
        for i in range(example_data.shape[0]):
            image_data = example_data[i].squeeze()
            plt.imsave(f'{example_path}/frame_{i}.png', image_data, cmap='gray')
    print(f"Saved to {save_folder}")


# Generate dataset
def generate_moving_mnist_binary(n_digit=1):
    mnist = datasets.MNIST(root='./data', train=True, download=True).data
    moving_mnist = generate_moving_mnist_multiple(mnist, num_samples=1000, n_digit=n_digit)
    moving_mnist = torch.squeeze(moving_mnist, 2)
    moving_mnist = torch.permute(moving_mnist, [1, 0, 2, 3])
    # print("Generated Moving MNIST dataset shape:", moving_mnist.shape)
    # print(torch.max(moving_mnist), torch.min(moving_mnist))
    np.save(f'output_dataset/moving_mnist_binary_{n_digit}_digit_1000.npy', moving_mnist.detach().numpy())

    one_time_example(torch.permute(moving_mnist, [1, 0, 2, 3]).detach().numpy(), f'dataset/moving_mnist_binary_{n_digit}_example', idx_list=range(10))


if __name__ == "__main__":
    generate_moving_mnist_binary(1)
    # generate_moving_mnist_binary(2)
    # generate_moving_mnist_binary(3)
