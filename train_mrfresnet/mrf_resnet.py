"""
The MRFResnet encoder-decoder network proposed in 
    M. Golbabaee, G. Bounincontri, C. Pirkl, M. Menzel, B. Menze, 
    M. Davies, and P. Gomez. "Compressive MRI quantification using convex
    spatiotemporal priors and deep auto-encoders." arXiv preprint arXiv:2001.08746 (2020).

    MATLAB: (c) 2018-2020 Mohammad Golbabaee, m.golbabaee@bath.ac.uk

    Adapted for Python: 2025 Ela Kanani, ela.kanani.21@ucl.ac.uk

    Classes for each element of the network. 
"""

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """
    Residual block to match RSB.m. Each conv layer expands to the number of
    hidden channels, and contracts back to the input size for the skip connection.
    """
    def __init__(self, datach, nbch2, depth, tag = None):
        """
        Parameters:
        -----------
            datach : int
                Number of input/output channels
            nbch2 : int
                Number of hidden channels (intermediate width)
            depth : int
                Number of conv layers in the block
 
        
        """
        super().__init__() 

        # number of layers is customisable
        layers = []

        # Inner convolutions: input channels to number of hidden channels 
        for layer in range(depth):
            layers.append(nn.Conv2d(datach, nbch2, kernel_size=1, padding=0))
            layers.append(nn.ReLU(inplace=True))

        # Outer convolution: back to the size of input channels 
        layers.append(nn.Conv2d(nbch2, datach, kernel_size=1, padding=0))

        # add sequence of layers 
        self.residual_layers = nn.Sequential(*layers) 
        
    def forward(self, x):
        return x + self.residual_layers(x)