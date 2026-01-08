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
    def __init__(self, datach, hidden_size, depth):
        """
        Parameters:
        -----------
            datach : int
                Number of input/output channels (= number of SVD components for MRF)
            hidden_size : int
                Number of hidden channels (intermediate width) = nbch2 in MATLAB implementation
            depth : int
                Number of conv layers in the block
 
        
        """
        super().__init__() 

        # number of layers is customisable. this is to build the residual layer path
        layers = []

        # Inner convolutions: input channels to number of hidden channels 
        # First inner convolution:
        layers.append(nn.Conv2d(datach, hidden_size, kernel_size=1, padding=0))
        layers.append(nn.ReLU(inplace=True))

        # Next inner convolutions when depth > 1
        for layer in range(1, depth):
            layers.append(nn.Conv2d(hidden_size, hidden_size, kernel_size=1, padding=0))
            layers.append(nn.ReLU(inplace=True))

        # Outer convolution: back to the size of input channels 
        layers.append(nn.Conv2d(hidden_size, datach, kernel_size=1, padding=0))

        # register all layers
        self.residual_layers = nn.Sequential(*layers) 
        
    def forward(self, x):
        return x + self.residual_layers(x) # add skip connection
    
class MRFResnetEncoder(nn.Module):
    """
    Encoder module to match MRFResnet.m. Contains Encoder network with residual blocks and
    skip connections between the blocks.

    Takes compressed fingerprints and estimates parameters (i.e. T1/T2)

    """

    def __init__(self, datach, hidden_size, depth_out, depth_int, num_params = 2):
        """
        Parameters:
        -----------
            datach : int
                Input/output channels (= number of SVD components for MRF)
            hidden_size : int
                Hidden layer width in residual blocks (= 10) = nbch2 in MATLAB implementation
            depth_out : int
                Number of residual blocks (= 6 in paper)
            depth_int : int
                Depth within each residual block (= 1 in paper)
            num_params : int
                Number of output parameters. Automatically set to 2 for T1/T2

            nbch1 : int
                Not used in python implementation. It is equal to datach in orignal MATLAB.
                It follows that the input/output channels should match.
        """
        super().__init__()

        # for tracking
        self.residual_blocks = nn.ModuleList() 
        self.relus = nn.ModuleList()

        # residual blocks
        for resBlock in range(depth_out): # creating the N residual blocks 
            self.residual_blocks.append(ResidualBlock(datach, hidden_size, depth_int))
            self.relus.append(nn.ReLU(inplace=True))

        # output/regression layer
        # compress output to number of measured parameters
        self.conv_out = nn.Conv2d(datach, num_params, kernel_size=1, padding=0)
        self.relu_out = nn.ReLU(inplace=True) # output relu

    def forward(self, x):
        # residual layers
        residual = x # skip connection initialised with input
        for i, (res_block, relu) in enumerate(zip(self.residual_blocks, self.relus)):
            # add skip connection to all blocks except for the first block
            if i == 0:
                x = res_block(x)
            else:
                x = res_block(x+residual)

        x = relu(x)
        # update skip connection output
        residual = x

        # regression layer
        out = self.conv_out(x)
        out = self.relu_out(out)

        return out
    
class MRFResnetDecoder(nn.Module):
    """
    Encoder module to match model_trainer.m decoder.
    
    Takes estimated parameters and generates clean fingerprints.
    """

    def __init__(self, datach = 10, input_channels = 2, hidden_size = 300):
        """
        Parameters:
        -----------
            datach : int
                Output channels (= number of SVD components for MRF)
            input_channels : int
                Equal to the number of parameters (i.e. 2 in original paper)
            hidden_size : int
                Hidden layer number of neurons (300 in MATLAB paper)
            output_channels : int
                Output channels (fingerprint width = 10)
        """

        super().__init__()
        # from parameter input to hidden layers
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=hidden_size, kernel_size=1, padding = 0)
        self.relu = nn.ReLU(inplace=True)

        # from hidden layers to compressed fingerprint size
        self.fc2 = nn.Conv2d(in_channels=hidden_size, out_channels=datach, kernel_size=1, padding=0)

    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)  
        out = self.fc2(out) 

        return out