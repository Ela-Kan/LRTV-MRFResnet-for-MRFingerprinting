"""
Main training script for the MRFResnet encoder-decoder, based on Train_MRFResnet.m

(c) 2018-2020 Mohammad Golbabaee, m.golbabaee@bath.ac.uk

Adapted for Python: 2025/2026 Ela Kanani, ela.kanani.21@ucl.ac.uk


TODO: add validation loss for improvement

"""

import torch
import numpy as np
import os
from datetime import datetime
from scipy import stats
import json

from data.prepare_data import prepare_data
from model_trainer import train_model
from utils import print_training_summary, plot_training_curves
                                   
def main():
    """
    Main training pipeline for MRFResNet.

    Returns:
    --------
        results : dict
        For post-processing/plotting. Keys in dictionary:
            * 'nn': Network metadata (normalisation factors etc.)
            * 'encoder_training_info': Encoder training history
            * 'decoder_training_info': Decoder training history
    """
    # ================ Set up ================
    # Set up device for training, options for cuda, apple silicon and cpu
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}\n")

    # Create output directories for trained model
    model_dir = './trained_models'
    os.makedirs(model_dir, exist_ok=True)

    # ================ Data preparation ================
    print("=" * 50)
    print("Data preparation & Augmentation")
    print("=" * 50)

    opt = {}
    opt["svd"] = 10
    dictionaryId = 'McGivneyFISP'
    training_data_folder = 'Dictionaries/Dictionary'+dictionaryId
    opt["foldername"] = training_data_folder
    train_x, train_y, nn = prepare_data(opt)

    print(f"Data loaded. train_x shape = {train_x.shape}. train_y shape: {train_y.shape}\n")


    # Noisy data augmentation for training the Encoder MRFResnetEncoder
    # Add noise to each fingerprint and then do dictionary search to see to which T1/T2 labels it matches best, then do the label correction.
    opt["augsize"] = 10 # number of noisy realisations of each figerprint (50 in the paper).
    X_aug = []
    Y_aug = []

    for i in range(opt["augsize"]):
        C = int(np.ceil(train_x.shape[0] / 4)) # batch size
        for j in range(4):
            batch_idx_start = j * C
            batch_idx_end = min((j + 1) * C, train_x.shape[0])
        
            # add gaussian noise to batch with sigma = 0.01
            batch_x = train_x[batch_idx_start:batch_idx_end, :] + 1e-2 * np.random.randn(batch_idx_end-batch_idx_start, train_x.shape[1])
            
            # normalise batch
            batch_x_norm = np.linalg.norm(batch_x, axis=1, keepdims=True)
            batch_x = batch_x/(batch_x_norm + 1e-10) # avoid division by 0

            # dictionary search
            similarities = train_x @ batch_x.T
            ind = np.argmax(similarities, axis=0)

            # correct labels according to search result
            batch_y = train_y[ind, :]

            X_aug.append(batch_x)
            Y_aug.append(batch_y)

    # convert to np array
    X_aug = np.concatenate(X_aug, axis=0)
    Y_aug = np.concatenate(Y_aug, axis=0)

    print(f"Training data augmented. X_aug shape = {X_aug.shape}. Y_aug shape: {Y_aug.shape}\n")

    # ================ Training data standardisation/normalisation ================
    print("=" * 50)
    print("Training data standardisation/normalisation")
    print("=" * 50)

    nn['sigma'] = X_aug.std(axis=0) # column std
    nn['tr_mu'] = X_aug.mean(axis=0) # column mean
    X_aug = stats.zscore(X_aug, ddof=1) # ddof set to match matlab implementation. save normalisation factors for future data

    print(f"Training data normalised.\n")

    # ================ Encoder training ================
    # model nn.Netbw in matlab: This is residual network MRFResnet which learns to embed Dictionary-Matching and find correct (T1,T2, etc) values given noisy fingerprints.
    print("=" * 50)
    print("Training Encoder...")
    print("=" * 50)

    # convert training data to pytorch format [batch, channels, 1, 1]
    X_encoder = torch.from_numpy(X_aug[:, :, None, None]).float()  
    Y_encoder = torch.from_numpy(Y_aug[:, :, None, None]).float()  

    encoder, encoder_training_info, encoder_timestamp = train_model(
            train_x=X_encoder,
            train_y=Y_encoder,
            modelname='Encoder',
            device=device,
            verbose=True
        )
    

    nn['encoder_training_info'] = encoder_training_info
    print("Finished training encoder.")

    # ================ Decoder training ================
    # model nn.bw in matlab: % This is a simple shallow network with one hiden layer which creates clean magnetic responses (fingerprints) given (T1,T2 etc)
    print("=" * 50)
    print("Preparing Decoder Training Data")
    print("=" * 50)

    X_decoder = train_y 
    Y_decoder = train_x* nn['normD'][:, np.newaxis] # undo dictionary normalisation

    print(f"Decoder training data prepared. X_decoder shape = {X_decoder.shape}. Y_decoder shape: {Y_decoder.shape}\n")

    print("=" * 50)
    print("Training Decoder...")
    print("=" * 50)

    decoder, decoder_training_info, decoder_timestamp = train_model(
        train_x=X_decoder,
        train_y=Y_decoder,
        modelname='Decoder',
        device=device,
        verbose=True
    )

    nn['decoder_training_info'] = decoder_training_info
    print("Finished training decoder.")

     # ================ Saving models and training info ================
    print("=" * 50)
    print("Saving final models and training info")
    print("=" * 50)

    
    model_dir = './trained_models'

    # save final encoder and decoder weights
    encoder_path = os.path.join(model_dir, f"encoder_{encoder_timestamp}.pth")
    decoder_path = os.path.join(model_dir, f"decoder_{decoder_timestamp}.pth")

    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)

    # save training info from models as human-readable json
    training_info = {
        'device': str(device),
        'sigma': nn['sigma'].tolist(),  
        'tr_mu': nn['tr_mu'].tolist(),
        'normD': nn['normD'].tolist(),
        'encoder_training_info': {
            'loss': encoder_training_info['loss'],
            'epoch': encoder_training_info['epoch'],
            'lr': encoder_training_info['lr']
        },
        'decoder_training_info': {
            'loss': decoder_training_info['loss'],
            'epoch': decoder_training_info['epoch'],
            'lr': decoder_training_info['lr']
        },
        'encoder_config': {
            'datach': 10,
            'hidden_size': 10,
            'depth_out': 6,
            'depth_int': 1,
            'num_params': 2
        },
        'decoder_config': {
            'datach': 10,
            'input_channels': 2,
            'hidden_size': 300
        }
    }
    training_info_path = f"{model_dir}/training_info_encoder{encoder_timestamp}_decoder{decoder_timestamp}.json"
    with open(training_info_path, 'w') as f:
        json.dump(training_info, f, sort_keys = True, ensure_ascii=False, indent=4)

    print(f"Saved models and metadata. Encoder Timestamp ID: {encoder_timestamp}. Decoder Timestamp ID: {decoder_timestamp}")

    return  {'nn' : nn, 'encoder_id' : encoder_timestamp, 'decoder_id' : decoder_timestamp, 'encoder_training_info': encoder_training_info, 'decoder_training_info': decoder_training_info}

if __name__ == '__main__':
    # train encoder and decoder
    results = main()
    
    # after training plots
    print_training_summary(results['encoder_training_info'])
    print_training_summary(results['decoder_training_info'])

    plot_training_curves(results['encoder_training_info'], results['encoder_id'])
    plot_training_curves(results['decoder_training_info'], results['decoder_id'])
    
    print("\n" + "=" * 50)
    print("Finished training and output report generation.")
    print("=" * 50)
