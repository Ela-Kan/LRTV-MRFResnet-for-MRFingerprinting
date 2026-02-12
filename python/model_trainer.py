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
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import os
from datetime import datetime


from model.mrf_resnet import MRFResnetEncoder, MRFResnetDecoder

def save_network(network, epoch, model_name, verbose= True, timestamp = None, save_dir = './trained_models'):
    """
    Saves model checkpoint at periodic intervals. Allows for retrieving prior models in case of
    overfitting.
    
    
    Based on https://niruhanv.medium.com/periodically-save-trained-neural-network-models-in-pytorch-f0837cc867e7
    
    Parameters:
    -----------
        network : nn.Module
            Model to save
        epoch : int
            Current epoch number
        model_name : str
            'Encoder' or 'Decoder'
        verbose : bool optional
            Print training progress
        timestamp : str
            Timestamp for checkpoint/model name (essentially an ID)
        save_dir : str
            Where to save the checkpoint
    
        
    """
    save_checkpoint_dir = f'{save_dir}/checkpoints'
    os.makedirs(save_checkpoint_dir, exist_ok=True)
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_filename = f"{model_name}_{timestamp}_epoch_{epoch}.pth"
    checkpoint_path = os.path.join(save_checkpoint_dir,save_filename)
    torch.save(network.state_dict(), checkpoint_path)
    if verbose:
        print(f"Saved {model_name} model at checkpoint: {checkpoint_path}")

def train_model(train_x, train_y, modelname, device = 'cpu', verbose = True, checkpoint_interval=10):
    """
    Train either Encoder or Decoder network. Matches model_trainer.m in matlab.
    Has checkpoint saving of models.
    
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
        checkpoint_interval : int optional
            Save checkpoint every N epochs 
    
    Returns:
    --------
        model : nn.Module
            Trained model
        training_history : dict
            Loss and learning rate history over epochs
        checkpoint_timestamp : str
            Time of first checkpoint, used as an ID for models
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

        #TODO: allow flexibility for more output parameters and model changes
        model = MRFResnetEncoder(
            datach=input_dim,          # Compressed fingerprint size
            hidden_size=10,     # Hidden channels in residual blocks (nbch2)
            depth_out=6,        # Number of residual blocks
            depth_int=1,        # Depth within each block
            num_params=output_dim        # Output: T1, T2
        ).to(device)

        max_epochs = 50 # 20 in paper
        batch_size = 100 # 500 in paper
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

        
        max_epochs = 100 # 20 in paper
        batch_size = 20 
        learning_rate = 0.01
        lr_decay_factor = 0.95
        lr_decay_period = 1
        beta1 = 0.95  # GradientDecayFactor 
        weight_decay = 0 # No L2 regularisation
        shuffle = True

    else:
       raise ValueError(f"Model: {modelname} does not exist. Check models.")

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

    
    # Split into training and validation sets
    dataset = TensorDataset(train_x, train_y)
    total_samps = len(dataset)
    train_samps = int(0.9*total_samps)
    val_samps = total_samps - train_samps
    split_seed = torch.Generator().manual_seed(13)
    train_dataset, val_dataset = random_split(dataset, [train_samps, val_samps], generator = split_seed)


    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Data Split: {train_samps} training samples, {val_samps} validation samples.")

    # Training loop
    # for tracking training
    training_history = {'model' : modelname,
                        'train_loss' : [],
                        'val_loss' : [],
                        'epoch' : [],
                        'lr' : []}
    checkpoint_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # track best model based on validation set
    best_val_loss = np.inf
    early_stopping_criteria = 15 # stop training if val doesn't improve for 15 epochs
    epochs_no_improvement = 0

    for epoch in range(max_epochs):
        # training 
        model.train()
        running_train_loss  = 0.0
        for batch_x, batch_y in train_dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad() # zero the parameter gradients
            # Forward pass
            predictions = model(batch_x)
            loss = loss_function(predictions, batch_y)

            # Weighted avg loss per epoch. (https://discuss.pytorch.org/t/plotting-loss-curve/42632/4)
            running_train_loss += loss.item() * batch_x.size(0)

            # Backward pass
            loss.backward()
            optimizer.step()
        # average loss for the epoch
        epoch_train_loss = running_train_loss / len(train_dataset)
        
        # validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for val_x, val_y in val_dataloader:
                val_x = val_x.to(device)
                val_y = val_y.to(device)
                val_preds = model(val_x)
                val_loss = loss_function(val_preds, val_y)
                running_val_loss += val_loss.item() * val_x.size(0)
            
        epoch_val_loss = running_val_loss / len(val_dataset)
        
        # logging
        training_history['train_loss'].append(epoch_train_loss)
        training_history['val_loss'].append(epoch_val_loss)
        training_history['epoch'].append(epoch + 1)
        current_lr = optimizer.param_groups[0]['lr']
        training_history['lr'].append(current_lr)

        # check if validation loss improved
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improvement = 0
            # Save best model
            best_model_path = os.path.join('./trained_models', f"{modelname}_{checkpoint_timestamp}_lowest_val_loss.pth")
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improvement += 1

        # early stopping
        if epochs_no_improvement >= early_stopping_criteria:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch+1}. No improvement in validation loss for {early_stopping_criteria} epochs.")
            break

        # step learning rate scheduler
        lr_scheduler.step()

        # periodic saving checkpoint for model
        if (epoch + 1) % checkpoint_interval == 0:
            save_network(model, epoch+1, modelname, verbose, checkpoint_timestamp)

        if verbose and (epoch + 1) % max(1, max_epochs // 10) == 0:
            print(f"Epoch [{epoch+1:3d}/{max_epochs}], Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}, LR: {current_lr:.6f}")

    if verbose:
        print(f"\nFinal Train loss: {training_history['train_loss'][-1]:.6f}, Final Val Loss:{training_history['val_loss'][-1]:.6f}, Best Val Loss: {best_val_loss:.6f} \n")

    # Load and return the best model (lowest validation loss)
    best_model_path = os.path.join('./trained_models', f"{modelname}_{checkpoint_timestamp}_lowest_val_loss.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        if verbose:
            print(f"Loaded best model from checkpoint: {best_model_path}\n")
    else:
        if verbose:
            print(f"Best model checkpoint not found at {best_model_path}. Returning final epoch model.\n")

    return model, training_history, checkpoint_timestamp