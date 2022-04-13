import torchvision.utils
import numpy as np
import matplotlib.pyplot as plt
from dataloader import get_lsun_dataloader
import torch


# This function takes as an input the images to reconstruct
# and the name of the model with which the reconstructions
# are performed
def to_img(x):
    x = x.clamp(0, 1)
    return x


def show_image(fname, img):
    img = to_img(img)
    npimg = img.numpy()
    plt.imsave(fname, np.transpose(npimg, (1, 2, 0)))


def visualise_output(fname, images, model, device):
    with torch.no_grad():
        images = images.to(device)
        images, _, _ = model(images)
        print(images)
        print(images.shape)
        images = images.cpu()
        images = to_img(images)
        np_imagegrid = torchvision.utils.make_grid(images[0:64], 8, 8).numpy()
        plt.imsave(fname, np.transpose(np_imagegrid, (1, 2, 0)))


def img_main(data_path, niter):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Todo: change path to local path to .mdb file
    _, test_loader = get_lsun_dataloader(data_path, batch_size=64)
    images, labels = iter(test_loader).next()

    # First visualise the original images
    # print('Original images')
    # show_image('original-images.png', torchvision.utils.make_grid(images[1:50], 10, 5))
    # plt.show()

    # Reconstruct and visualise the images using the vae
    print('VAE reconstruction:')
    model_num = ((int(niter)-1) // 10) * 10

    vae = torch.load( f"vae_{model_num}.pt")
    visualise_output(f"vae_recons_{model_num}.png", images, vae, device)


if __name__ == "__main__":
    img_main('/root/data', 90)