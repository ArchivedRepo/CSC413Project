import numpy as np
import torch
from VAE import VariationalAutoencoder, vae_loss
from dataloader import get_lsun_dataloader
import json
from VAE_imsave import *
import matplotlib.pyplot as plt

latent_dims = 20
num_epochs = 100
batch_size = 512
capacity = 64
input_dim = 64
learning_rate = 1e-3
use_gpu = True
log_step = 10
losses = {"iteration": [], "loss": []}
model_save_step = 10
max_epochs_stop = 5


def train(datapath, num_epochs=num_epochs):

    vae = VariationalAutoencoder(capacity, latent_dims, input_dim)

    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    vae = vae.to(device)

    num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print('Number of parameters: %d' % num_params)

    optimizer = torch.optim.Adam(params=vae.parameters(), lr=learning_rate, weight_decay=1e-5)

    # set to training mode
    vae.train()

    train_loss_avg = []
    # Todo: change path to local path to .mdb file
    train_loader, test_loader = get_lsun_dataloader(datapath, batch_size=batch_size)

    # early stopping
    count_stop = 0
    min_loss_each = np.inf

    print('Training ...')
    for epoch in range(num_epochs):
        train_loss_avg.append(0)
        num_batches = 0

        for image_batch, _ in train_loader:

            image_batch = image_batch.to(device)

            # vae reconstruction
            image_batch_recon, latent_mu, latent_logvar = vae(image_batch)

            # reconstruction error
            loss = vae_loss(image_batch_recon, image_batch, latent_mu, latent_logvar)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()

            # one step of the optimizer (using the gradients from backpropagation)
            optimizer.step()

            train_loss_avg[-1] += loss.item()

            if num_batches % log_step == 0:
                losses['iteration'].append(num_batches)
                losses['loss'].append(loss.item())
                print('Iteration [{:4d}/{:4d}] | train_loss: {:6.4f}'.format(
                    num_batches, 100, loss.item()))
            num_batches += 1

        train_loss_avg[-1] /= num_batches

        vae.eval()

        with torch.no_grad():

            # sample latent vectors from the normal distribution
            latent = torch.randn(batch_size, latent_dims, device=device)

            # reconstruct images from the latent vectors
            img_recon = vae.decoder(latent)
            img_recon = img_recon.cpu()

            fig, ax = plt.subplots(figsize=(5, 5))
            show_image(f"vae_sample_{epoch}.png",torchvision.utils.make_grid(img_recon.data[:100], 10, 5))

        print('Epoch [%d / %d] average reconstruction error: %f' % (epoch + 1, num_epochs, train_loss_avg[-1]))
        if epoch % model_save_step == 0 and epoch != 0:
            print(f"Saving model at step {epoch}")
            torch.save(vae, f"vae_{epoch}.pt")

        if min_loss_each > train_loss_avg[-1]:
            min_loss_each = train_loss_avg[-1]
            count_stop = 0
        else:
            count_stop += 1
            print("count stop " + str(count_stop))
            if count_stop >= max_epochs_stop:
                print(f'\n Early Stopping !')
                break

    plt.plot(train_loss_avg)
    plt.title("VAE Training Curve")
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.savefig('vae_loss_curve.png')

    with open('losses.log', 'w') as ptr:
        json.dump(losses, ptr)