from torch import nn
import torch.nn.utils.spectral_norm as spectral_norm

import numpy as np


class Discriminator(nn.Module):
    """
    Discriminator class for DCGAN.

    :param image_size: size of the input image (assumed to be square). Must be a power of 2
    :param feat_map_size: number of feature maps in the first layer of the discriminator
    :param num_channels: number of channels in the input image
    :param dropout: dropout probability
    """

    def __init__(self, image_size, feat_map_size, num_channels, dropout=0):
        super(Discriminator, self).__init__()

        blocks = []

        prev_dim = num_channels
        for i in range(int(np.log2(image_size)) - 2):
            blocks.append(
                self._get_conv_block(
                    in_channels=prev_dim,
                    out_channels=feat_map_size * (2**i),
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    dropout=dropout,
                    activation=nn.LeakyReLU(0.2, inplace=True),
                    batch_norm = False if i==0 else True
                )
            )
            prev_dim = feat_map_size * (2**i)

        blocks.append(
            self._get_conv_block(
                in_channels=prev_dim,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                dropout=0,
                activation=nn.Sigmoid(),
                batch_norm=False
            )
        )

        self.model = nn.Sequential(*blocks)

    def _get_conv_block(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dropout,
        activation,
        batch_norm = True
    ):
        return nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=not batch_norm,
                )
            ),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
            activation,
        )

    def forward(self, images):
        return self.model(images)
