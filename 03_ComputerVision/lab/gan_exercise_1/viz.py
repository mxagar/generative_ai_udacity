from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

# Visualize the output tensor as a grayscale image
def visualize_batch(batch):
    b = batch.detach().cpu()
    fig, sub = plt.subplots(dpi=150)
    sub.imshow(
        np.transpose(
            make_grid(
                b, 
                padding=0,
                normalize=True
            ).cpu(),
            (1,2,0)
        )
    )
    _ = sub.axis("off")


def training_tracking(D_losses, G_losses, D_acc, fake_data):

    fig = plt.figure(dpi=150)

    gs = gridspec.GridSpec(2, 8)

    # Create subplots
    ax_a = fig.add_subplot(gs[0, :3])  # Top-left subplot
    ax_b = fig.add_subplot(gs[1, :3])  # Bottom-left subplot
    ax_c = fig.add_subplot(gs[:, 4:])  # Right subplot spanning both rows

    subs = [ax_a, ax_b, ax_c]

    # Losses
    subs[0].plot(D_losses, label="Discriminator")
    subs[0].plot(G_losses, label="Generator")
    subs[0].legend()
    subs[0].set_ylabel("Loss")

    # Accuracy
    subs[1].plot(D_acc)
    subs[1].set_ylabel("D accuracy")

    # Examples of generated images
    subs[2].imshow(
        np.transpose(
            make_grid(
                fake_data.detach().cpu(), padding=0, normalize=True, nrow=4
            ).cpu(),
            (1, 2, 0),
        )
    )
    subs[2].axis("off")
    fig.tight_layout()
    
    return fig
