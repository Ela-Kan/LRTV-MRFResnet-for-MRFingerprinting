"""
Training functions for MRFResnet. Based on model_trainer.m

Follows the MRFResnet encoder-decoder model proposed in 

     M. Golbabaee, G. Bounincontri, C. Pirkl, M. Menzel, B. Menze, 
     M. Davies, and P. Gomez. "Compressive MRI quantification using convex
     spatiotemporal priors and deep auto-encoders." arXiv preprint arXiv:2001.08746 (2020).

 (c) 2018-2020 Mohammad Golbabaee, m.golbabaee@bath.ac.uk

 Adapted for Python: 2025 Ela Kanani, ela.kanani.21@ucl.ac.uk

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from model.mrf_resnet import MRFResnetEncoder, MRFResnetDecoder

def train_model(train_x, train_y, modelname, device = 'cpu', verbose = True):
    """
    Train either Encoder or Decoder network. Matches model_trainer.m in matlab
    
    Parameters:
    -----------
        train_x : torch.Tensor or np.ndarray
            Input data [batch, channels, height, width]
        train_y : torch.Tensor or np.ndarray
            Target data [batch, channels, height, width]
        model_name : str
            'Encoder' or 'Decoder'
        device : str
            'cpu' or 'cuda' or 'mps'
        verbose : bool optional
            Print training progress
    
    Returns:
    --------
        model : nn.Module
            Trained model
        training_history : dict
            Loss history
    """
    
    # convert training data to tensors if necessary
    if isinstance(train_x, np.ndarray):
        train_x = torch.from_numpy(train_x).float()
    if isinstance(train_y, np.ndarray):
        train_y = torch.from_numpy(train_y).float()

    # ensure correct shape for 1x1 convolutions. 
    # pytorch conv2d requires (batch, channel, height, width), so if the input is (batch, channel), it should be unsqueezed for
    # height = 1, width = 1.
    if train_x.dim() == 2:
        train_x = train_x.unsqueeze(-1).unsqueeze(-1)
    if train_y.dim() == 2:
        train_y = train_y.unsqueeze(-1).unsqueeze(-1)

    # dimensions of data
    input_dim = train_x.shape[1] # channels
    output_dim = train_y.shape[1] # num params

    if modelname == 'Encoder':
        print("=" * 50)
        print("Training MRFResnet Encoder")
        print("=" * 50)

        #TODO: allow flexibility for more output parameters
        model = MRFResnetEncoder(
            datach=input_dim,          # Compressed fingerprint size
            hidden_size=10,     # Hidden channels in residual blocks (nbch2)
            depth_out=6,        # Number of residual blocks
            depth_int=1,        # Depth within each block
            num_params=output_dim        # Output: T1, T2
        ).to(device)

        max_epochs = 50
        batch_size = 100
        learning_rate = 0.01
        lr_decay_factor = 0.8
        lr_decay_period = 1
        beta1 = 0.95  # = GradientDecayFactor 
        weight_decay = 0 # No L2 regularisation
        shuffle = True # shuffle data each epoch

    elif modelname == 'Decoder':
        print("=" * 50)
        print("Training MRFResnet Decoder")
        print("=" * 50)

        #TODO: allow flexibility for more output parameters
        model = MRFResnetDecoder(
            datach=output_dim,          # Output fingerprint width
            input_channels=input_dim,   # Input: T1, T2
            hidden_size=300     # Hidden layer neurons
        ).to(device)

        
        max_epochs = 100
        batch_size = 20
        learning_rate = 0.01
        lr_decay_factor = 0.95
        lr_decay_period = 1
        beta1 = 0.95  # GradientDecayFactor 
        weight_decay = 0 # No L2 regularisation
        shuffle = True

    else:
       raise ValueError(f"Model: {modelname} does not exist. Check models.'")

    # Set up optimiser and loss function
    optimizer = optim.Adam(model.parameters(), 
                           lr = learning_rate,
                           betas=(beta1, 0.999),
                           weight_decay=weight_decay)
    
    # adjust learning rate
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
        step_size=lr_decay_period,
        gamma = lr_decay_factor
    )

    loss_function = nn.MSELoss()

    # Create DataLoader
    dataset = TensorDataset(train_x, train_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    # Training loop

    # for tracking training
    training_history = {'loss' : [],
                        'epoch' : [],
                        'lr' : []}
    
    model.train()

    for epoch in range(max_epochs):
        running_loss  = 0.0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad() # zero the parameter gradients
            # Forward pass
            predictions = model(batch_x)
            loss = loss_function(predictions, batch_y)
            # Backward pass
            loss.backward()
            optimizer.step()

        # Weighted avg loss per epoch. (https://discuss.pytorch.org/t/plotting-loss-curve/42632/4)
        running_loss += loss.item() * batch_x.size(0)
        epoch_loss = running_loss / len(dataset)
        current_lr = optimizer.param_groups[0]['lr']

        training_history['loss'].append(epoch_loss)
        training_history['epoch'].append(epoch + 1)
        training_history['lr'].append(current_lr)

        # step learning rate scheduler
        lr_scheduler.step()

        if verbose and (epoch + 1) % max(1, max_epochs // 10) == 0:
            print(f"Epoch [{epoch+1:3d}/{max_epochs}], Loss: {epoch_loss:.6f}, LR: {current_lr:.6f}")

    if verbose:
        print(f"\nFinal loss: {training_history['loss'][-1]:.6f}\n")

    return model, training_history