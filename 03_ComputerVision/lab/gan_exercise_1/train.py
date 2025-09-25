import os
import random
import argparse
from pathlib import Path
from matplotlib import gridspec, pyplot as plt

from torch import nn, optim
import torch
import numpy as np
import tqdm

from data import get_dataloader
from viz import training_tracking
from diff_augment import DiffAugment
from generator import Generator
from discriminator import Discriminator

from torchvision.utils import make_grid



manualSeed = 42
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results


def get_optimizer_and_loss(D, G, lr, beta1):
    """
    Returns the optimizer to be used for training.

    :param D: discriminator network
    :param G: generator network
    :param lr: learning rate
    :param beta1: beta1 parameter for Adam optimizer
    """
    # Setup optimization
    # Initialize the ``BCELoss`` function
    criterion = nn.BCELoss()

    # Setup Adam optimizers for both G and D
    D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
    G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))

    return D_optimizer, G_optimizer, criterion


def weights_init(m):
    """
    Initialize Conv layers and BatchNorm from a normal distribution with mean 0 and std 0.02.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def get_positive_labels(size, device, smoothing=True, random_flip=0.05):
    if smoothing:
        # Random positive numbers between 0.8 and 1.2 (label smoothing)
        labels = 0.8 + 0.4 * torch.rand(size, device=device)
    else:
        labels = torch.full((size,), 1.0, device=device)

    if random_flip > 0:
        # Let's flip some of the labels to make it slightly harder for the discriminator
        num_to_flip = int(random_flip * labels.size(0))

        # Get random indices and set the first "num_to_flip" of them to 0
        indices = torch.randperm(labels.size(0))[:num_to_flip]
        labels[indices] = 0

    return labels


def get_negative_labels(size, device):
    return torch.full((size,), 0.0, device=device)


def train(
    data_root_path: Path,
    image_size: int,
    num_channels: int,
    latent_dimension: int,
    batch_size: int,
    lr: float,
    beta1: float,
    D_feat_map_size: int,
    G_feat_map_size: int,
    n_epochs: int,
    policy: str="color,translation,cutout",
    smoothing: bool=True,
    random_flip: float=0.05,
    gradient_clipping_value: float=0,
    jupyter: bool=False,
    output_dir: Path="training_results",
    save_iter: int=10,
    D_dropout: float=0.2
):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Let's generate a latent vector of fixed noise to visualize the progression
    # of the Generator
    fixed_noise = torch.randn(16, latent_dimension, 1, 1, device=device)

    # Get dataloader
    dataloader = get_dataloader(data_root_path, image_size, batch_size)

    # Setup model and optimizer
    G = Generator(image_size, latent_dimension, G_feat_map_size, num_channels)
    D = Discriminator(image_size, D_feat_map_size, num_channels, dropout=D_dropout)

    D_optimizer, G_optimizer, criterion = get_optimizer_and_loss(D, G, lr, beta1)

    D.to(device)
    D.apply(weights_init)
    G.to(device)
    G.apply(weights_init)

    # Lists to keep track of progress
    G_losses = []
    D_losses = []
    D_acc = []

    print("Training is starting...")

    # For each epoch
    for epoch in range(n_epochs):
        # Accumulate the losses for each batch in these lists
        batch_G_losses = []
        batch_D_losses = []
        # Accumulate the accuracy for each batch in this list
        batch_D_accuracy = []

        # We will train the Discriminator on a batch that is made by
        # half real data and half fake data. Then we will train the
        # Generator on a batch of only fake data.
        # Remember that we have configured the dataloader to return
        # batches of half batch_size, so the total batch size that the
        # Discriminator sees is batch_size (half real and half fake data)
        # just like the Generator

        for data in tqdm.tqdm(dataloader):
            
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            D.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            
            # label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Random positive numbers between 0.8 and 1.2 (label smoothing)
            label = 0.8 + 0.4 * torch.rand(b_size, device=device)
            
            # Let's flip 10% of the labels to make it slightly harder for the discriminator
            num_to_flip = int(0.05 * label.size(0))

            # Get random indices and set the first "num_to_flip" of them to 0
            indices = torch.randperm(label.size(0))[:num_to_flip]
            label[indices] = 0
            
            # Forward pass real batch through D
            output = D(DiffAugment(real_cpu, policy=policy)).view(-1)
            
            # Accuracy for positive
            acc_pos = (output > 0.5).sum() / output.size(0)
            
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, latent_dimension, 1, 1, device=device)
            # Generate fake image batch with G
            fake = G(noise)
            label.fill_(0.0)

            # Classify all fake batch with D
            output = D(DiffAugment(fake, policy=policy).detach()).view(-1)
            
            # Accuracy for negative
            acc_neg = (output < 0.5).sum() / output.size(0)
            
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Gradient clipping
            #nn.utils.clip_grad_value_(D.parameters(), clip_value=1.0)
            # Update D
            D_optimizer.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            G.zero_grad()
            label.fill_(1.0)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = D(DiffAugment(fake, policy=policy)).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Gradient clipping
            #nn.utils.clip_grad_value_(G.parameters(), clip_value=1.0)
            # Update G
            G_optimizer.step()

            # Output training stats
            # if i % 50 == 0:
            #     print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
            #           % (epoch, n_epochs, i, len(dataloader),
            #              errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
            # Save Losses for plotting later
            batch_G_losses.append(errG.item())
            batch_D_losses.append(errD.item())
            batch_D_accuracy.append((0.5 * (acc_pos + acc_neg)).item())

        G_losses.append(np.mean(batch_G_losses))
        D_losses.append(np.mean(batch_D_losses))
        D_acc.append(np.mean(batch_D_accuracy))

        if not jupyter:
            # Print losses and accuracy
            print(f"Epoch {epoch+1} / {n_epochs}")
            print(
                f"G loss: {G_losses[-1]:8.2f} | D loss: {D_losses[-1]:8.2f} | D accuracy: {D_acc[-1]:4.3f}"
            )
        
        if epoch % 50 == 0:
            with torch.no_grad():
                fake = G(fixed_noise).detach().cpu()

            #clear_output(wait=True)
            fig = plt.figure(dpi=150)

            gs = gridspec.GridSpec(2, 8)

            # Create subplots
            ax_a = fig.add_subplot(gs[0, :3])  # Top-left subplot
            ax_b = fig.add_subplot(gs[1, :3])  # Bottom-left subplot
            ax_c = fig.add_subplot(gs[:, 4:])  # Right subplot spanning both rows
            
            subs = [ax_a, ax_b, ax_c]

            subs[0].plot(G_losses, label="Generator")
            subs[0].plot(D_losses, label="Discriminator")
            subs[0].legend()
            subs[0].set_ylabel("Loss")
    #         subs[0].set_title("Losses")
            
            subs[1].plot(D_acc)
            subs[1].set_ylabel("D accuracy")

            # Now plot the accuracy
            # subs[1].plot([D_x, D_G_z1, D_G_z2])

            subs[2].imshow(
                np.transpose(
                    make_grid(
                        fake.detach().cpu(), 
                        padding=2,
                        normalize=True,
                        nrow=4
                    ).cpu(),
                    (1,2,0)
                )
            )
            subs[2].axis("off")
            fig.tight_layout()
            plt.show()

            os.makedirs("ff", exist_ok=True)
            fig.savefig(f"ff/epoch_{epoch}.png", bbox_inches='tight')

        # if epoch % save_iter == 0:
        #     with torch.no_grad():
        #         fake_viz_data = G(fixed_noise).detach().cpu()

        #     if jupyter:
        #         from IPython.display import clear_output
        #         import matplotlib.pyplot as plt

        #         clear_output(wait=True)

        #         fig = training_tracking(D_losses, G_losses, D_acc, fake_viz_data)

        #         plt.show()

        #     else:
        #         fig = training_tracking(D_losses, G_losses, D_acc, fake_viz_data)

        #     output_dir.mkdir(parents=True, exist_ok=True)
        #     fig.savefig(
        #         output_dir / f"training_tracking_{epoch}.png", bbox_inches="tight"
        #     )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training script")

    parser.add_argument("--data-root-path", type=Path, required=True, help="Path to the data root")
    parser.add_argument("--image-size", type=int, required=True, help="Size of the image")
    parser.add_argument("--num-channels", type=int, required=True, help="Number of channels in the image")
    parser.add_argument("--latent-dimension", type=int, required=True, help="Latent dimension size")
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size for training")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--beta1", type=float, required=True, help="Beta1 value for the Adam optimizer")
    parser.add_argument("--D-feat-map-size", type=int, required=True, help="Feature map size for Discriminator")
    parser.add_argument("--G-feat-map-size", type=int, required=True, help="Feature map size for Generator")
    parser.add_argument("--n-epochs", type=int, required=True, help="Number of epochs for training")

    # Default parameters
    parser.add_argument("--D-dropout", type=float, required=False, help="Dropout for the Discriminator", default=0)
    parser.add_argument("--policy", type=str, default="color,translation,cutout", help="Policy types")
    parser.add_argument("--smoothing", action='store_true', help="Enable label smoothing")
    parser.add_argument("--random-flip", type=float, default=0.05, help="Random flip probability")
    parser.add_argument("--gradient-clipping-value", type=float, default=0, help="Gradient clipping value")
    parser.add_argument("--jupyter", action='store_true', help="Running in Jupyter environment")
    parser.add_argument("--output-dir", type=Path, default="training_results", help="Directory to save output results")
    parser.add_argument("--save-iter", type=int, default=10, help="Frequency of saving results")

    args = parser.parse_args()
    
    print(vars(args))

    train(**vars(args))
