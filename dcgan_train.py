# *****************
# Training module for DCGAN
# *****************
import os
import torch
from torch.autograd import Variable
from dcgan import Generator, Discriminator
from dataloader import get_lsun_dataloader
import json
import matplotlib.pyplot as plt
import numpy as np
import imageio

generator_dim = 32
discriminator_dim = 64
latent_dim = 21       # noise size
learning_rate = 1e-3
weight_decay=1e-5
losses = {"iteration": [], "loss": []}
model_save_step = 10
batch_size = 64
d_train_iters = 1
train_iters = 1000
use_gpu = True
model_save_step = 100
log_step = 10
output_dir = "."
beta1 = 0.5
beta2 = 0.999

def create_model():
    """Builds the generators and discriminators.
    """
    ### GAN
    G = Generator(latent_dim=latent_dim, generator_dim=generator_dim)
    D = Discriminator(discriminator_dim=discriminator_dim)

    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    G = G.to(device)
    D = D.to(device)

    return G, D

def to_var(tensor, cuda=True):
    """Wraps a Tensor in a Variable, optionally placing it on the GPU.

        Arguments:
            tensor: A Tensor object.
            cuda: A boolean flag indicating whether to use the GPU.

        Returns:
            A Variable object, on the GPU if cuda==True.
    """
    if cuda:
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)


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

    path = os.path.join(output_dir, 'dcgan-{:06d}.png'.format(iteration))
    imageio.imwrite(path, grid)
    print('Saved {}'.format(path))


def sample_noise(batch_size, dim):
    """
    Generate a PyTorch Tensor of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Tensor of shape (batch_size, dim, 1, 1) containing uniform
      random noise in the range (-1, 1).
    """
    return to_var(torch.rand(batch_size, dim) * 2 - 1).unsqueeze(2).unsqueeze(3)


def train_dcgan(dataloader, test_dataloader):
    """Runs the training loop.
    """

    # Create generators and discriminators
    G, D = create_model()

    # Create optimizers for the generators and discriminators
    g_optimizer = torch.optim.Adam(G.parameters(), learning_rate, [beta1, beta2])
    d_optimizer = torch.optim.Adam(D.parameters(), learning_rate * 2., [beta1, beta2])

    train_iter = iter(dataloader)

    iter_per_epoch = len(train_iter)
    total_train_iters = train_iters

    losses = {"iteration": [], "D_fake_loss": [], "D_real_loss": [], "G_loss": []}
    # Sample a fixed noise
    fixed_noise = sample_noise(batch_size, latent_dim)

    try:
        for iteration in range(1, train_iters + 1):

            # Reset data_iter for each epoch
            if iteration % iter_per_epoch == 0:
                train_iter = iter(dataloader)

            real_images, real_labels = train_iter.next()
            real_images, real_labels = to_var(real_images), to_var(real_labels).long().squeeze()

            for d_i in range(d_train_iters):
                d_optimizer.zero_grad()

                # 1. Compute the discriminator loss on real images
                D_real_loss = torch.mean((D(real_images) - 1) ** 2) / 2

                # 2. Sample noise
                noise = sample_noise(real_images.shape[0], latent_dim)

                # 3. Generate fake images from the noise
                fake_images = G(noise)        

                # 4. Compute the discriminator loss on the fake images
                D_fake_loss = torch.mean(D(fake_images) ** 2) / 2

                # --------------------------
                # 5. Compute the total discriminator loss
                D_total_loss = D_real_loss + D_fake_loss

                D_total_loss.backward()
                d_optimizer.step()

            ###########################################
            ###          TRAIN THE GENERATOR        ###
            ###########################################

            g_optimizer.zero_grad()

            # 1. Sample noise
            noise = sample_noise(real_images.shape[0], latent_dim)

            # 2. Generate fake images from the noise
            fake_images = G(noise)

            # 3. Compute the generator loss
            G_loss = torch.mean((D(fake_images) - 1) ** 2)

            G_loss.backward()
            g_optimizer.step()

            # Print the log info
            if iteration % log_step == 0:
                losses['iteration'].append(iteration)
                losses['D_real_loss'].append(D_real_loss.item())
                losses['D_fake_loss'].append(D_fake_loss.item())
                losses['G_loss'].append(G_loss.item())
                print('Iteration [{:4d}/{:4d}] | D_real_loss: {:6.4f} | D_fake_loss: {:6.4f} | G_loss: {:6.4f}'.format(
                    iteration, total_train_iters, D_real_loss.item(), D_fake_loss.item(), G_loss.item()))

            # Save the model parameters
            if iteration % model_save_step == 0 and iteration != 0:
                print(f"Saving model at step {iteration}")
                torch.save(G, f"G_{iteration}.pt")
                torch.save(D, f"D_{iteration}.pt")
                gan_save_samples(G, fixed_noise, iteration)

    except KeyboardInterrupt:
        print('Exiting early from training.')
        return G, D

    plt.figure()
    plt.plot(losses['iteration'], losses['D_real_loss'], label='D_real')
    plt.plot(losses['iteration'], losses['D_fake_loss'], label='D_fake')
    plt.plot(losses['iteration'], losses['G_loss'], label='G')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'losses.png'))
    plt.close()
    return G, D


if __name__ == "__main__":
    train_loader, test_loader = get_lsun_dataloader('/root/data', batch_size=128)
    G, D = train_dcgan(train_loader, test_loader)