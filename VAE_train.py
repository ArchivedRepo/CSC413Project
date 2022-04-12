import torch
from VAE import VariationalAutoencoder, vae_loss
from dataloader import get_lsun_dataloader


latent_dims = 20
num_epochs = 3
batch_size = 64
capacity = 64
learning_rate = 1e-3
use_gpu = True
log_step = 10
losses = {"iteration": [], "loss": []}

vae = VariationalAutoencoder(capacity, latent_dims)

device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
vae = vae.to(device)

num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
print('Number of parameters: %d' % num_params)

optimizer = torch.optim.Adam(params=vae.parameters(), lr=learning_rate, weight_decay=1e-5)

# set to training mode
vae.train()

train_loss_avg = []
train_loader, test_loader = get_lsun_dataloader('/root/data')

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

        # one step of the optmizer (using the gradients from backpropagation)
        optimizer.step()

        train_loss_avg[-1] += loss.item()

        if num_batches % log_step == 0:
            losses['iteration'].append(num_batches)
            losses['loss'].append(loss.item())
            print('Iteration [{:4d}/{:4d}] | train_loss: {:6.4f}'.format(
                num_batches, 100, loss.item()))
        num_batches += 1

    train_loss_avg[-1] /= num_batches
    print('Epoch [%d / %d] average reconstruction error: %f' % (epoch + 1, num_epochs, train_loss_avg[-1]))