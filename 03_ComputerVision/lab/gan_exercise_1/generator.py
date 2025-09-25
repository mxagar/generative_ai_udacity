from torch import nn
import numpy as np


class Generator(nn.Module):
    """
    Generator class for DCGAN.

    :param image_size: size of the input image (assumed to be square). Must be a power of 2
    :param latent_dimension: dimension of the latent space
    :param feat_map_size: number of feature maps in the last layer of the generator
    :param num_channels: number of channels in the input image
    """
    def __init__(self, image_size, latent_dimension, feat_map_size, num_channels):
        
        super(Generator, self).__init__()

        # The following defines the architecture in a way that automatically
        # scales the number of blocks depending on the size of the input image

        # Number of blocks between the first and the last (excluded)
        n_blocks = int(np.log2(image_size)) - 3

        # Initial multiplicative factor for the number of feature maps
        factor = 2**(n_blocks)

        # The first block takes us from the latent space to the feature space with a
        # 4x4 kernel with stride 1 and no padding
        blocks = [
            self._get_transpconv_block(latent_dimension, feat_map_size*factor, 4, 1, 0, nn.LeakyReLU(0.2))
        ]

        # The following blocks are transposed convolutional layers with stride 2 and
        # kernel size 4x4. Every block halves the number of feature maps but double the
        # size of the image (upsampling)
        # (NOTE that we loop in reverse order)
        prev_dim = feat_map_size*factor
        for f in range(int(np.log2(factor)-1), -1, -1):
            blocks.append(
                self._get_transpconv_block(prev_dim, feat_map_size*2**f, 4, 2, 1, nn.LeakyReLU(0.2))
            )
            prev_dim = feat_map_size*2**f

        # Add last layer
        blocks.append(
            self._get_transpconv_block(feat_map_size, num_channels, 4, 2, 1, nn.Tanh(), batch_norm=False)
        )

        self.model = nn.Sequential(
            *blocks
        )
    
    def _get_transpconv_block(self, in_channels, out_channels, kernel_size, stride, padding, activation, batch_norm=True):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, 
                out_channels, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding
        ),
            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
            activation
        )

    def forward(self, latents):

        return self.model(latents)
