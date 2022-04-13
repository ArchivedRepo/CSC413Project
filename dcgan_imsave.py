import torchvision.utils
import numpy as np
import matplotlib.pyplot as plt
from dataloader import get_lsun_dataloader
import torch
import imageio
import os
from dcgan_train import *


# # This function takes as an input the images to reconstruct
# # and the name of the model with which the reconstructions
# # are performed
# def to_img(x):
#     x = x.clamp(0, 1)
#     return x
#
#
# def show_image(fname, img):
#     img = to_img(img)
#     npimg = img.numpy()
#     plt.imsave(fname, np.transpose(npimg, (1, 2, 0)))
#
#
# def visualise_output(fname, images, model, device):
#     with torch.no_grad():
#         images = images.to(device)
#         images = model(images)
#         print(images.shape)
#         images = images.cpu()
#         images = to_img(images)
#         np_imagegrid = torchvision.utils.make_grid(images[1:50], 10, 5).numpy()
#         plt.imsave(fname, np.transpose(np_imagegrid, (1, 2, 0)))
#
#
# def img_main(data_path, niter):
#
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     # Todo: change path to local path to .mdb file
#     _, test_loader = get_lsun_dataloader(data_path, batch_size=64)
#     images, labels = iter(test_loader).next()
#
#     # First visualise the original images
#     # print('Original images')
#     # show_image('original-images.png', torchvision.utils.make_grid(images[1:50], 10, 5))
#     # plt.show()
#
#     # Reconstruct and visualise the images using the vae
#     print('Dcgan reconstruction:')
#
#
#     model_num = (int(niter) // 10) * 10
#
#     dcgan = torch.load( f"G_{model_num}.pt")
#     visualise_output(f"dcgan_recons_{model_num}.png", images, dcgan, device)
#
# img_main('/media/anna/54F8F2E0F8F2BF74/CSC413Project/data/sheep', 100)


def create_image_grid(array, ncols=None):
    """
    """
    num_images, channels, cell_h, cell_w = array.shape
    if not ncols:
        ncols = int(np.sqrt(num_images))
    nrows = int(np.math.floor(num_images / float(ncols)))
    result = np.zeros((cell_h * nrows, cell_w * ncols, channels), dtype=array.dtype)
    for i in range(0, nrows):
        for j in range(0, ncols):
            result[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w, :] = array[i * ncols + j].transpose(1, 2,
                                                                                                                 0)

    if channels == 1:
        result = result.squeeze()
    return result


def to_data(x):
    """Converts variable to numpy."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()


def gan_save_samples(G, fixed_noise, iteration):
    generated_images = G(fixed_noise)
    generated_images = to_data(generated_images)

    grid = create_image_grid(generated_images)

    path = os.path.join(output_dir, 'sample-{:06d}.png'.format(iteration))
    imageio.imwrite(path, grid)
    print('Saved {}'.format(path))


if __name__ == "__main__":
    G = torch.load( f"G_100.pt")

    fixed_noise = sample_noise(batch_size, latent_dim)
    print(fixed_noise.shape)
    gan_save_samples(G, fixed_noise, train_iters)