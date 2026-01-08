import numpy as np
import h5py
import os

"""
Data loading for training MRFResNet.

Based on (c) 2018-2020 Mohammad Golbabaee, m.golbabaee@bath.ac.uk

Converted to Python and adapted: 2025 Ela Kanani ela.kanani.21@ucl.ac.uk

"""

def load_dic_svd(data_folder):
    """
    Loads in MRF dictionary with lookup table, and pre-computed SVD decomposition of dictionary.
    
    Parameters:
    -----------
    data_folder : str
        Folder containing dictionary (dictionary.h5) and SVD decomposition (SVD.h5). If no SVD file is found,
        SVD will be performed.
    
    Returns:
    --------
    dictionary : array [nSamps, nDynamics]
        The complex-valued MRF dictionary
    lookup_table : array [nSamps, nParams]
        Parameter lookup table 
    U : array [nSamps, nkComponents]
        Left singular vectors from SVD
    s : array [nkComponents]
        Singular values from SVD
    Vh : array [nkComponents, nDynamics]
        Right singular vectors (conjugate transpose) from SVD

    NOTE: In np.linalg.svd U, diagonal of S and V^H are returned. To be equivalent to MATLAB, S will need
    to be reconstructed, and V = Vh.conj().T (no conj needed with real numbers)
    
    """

    with h5py.File(data_folder+'/dictionary.h5', 'r') as f:
        dictionary = f['dictionary'][:]
        lookup_table = f['parameters'][:]

    # collapse the dictionary so that is of shape [nSamps, nDynamics]--i.e. add the real and imaginary channels together if they are not already
    if not np.iscomplexobj(dictionary[0]):
        dictionary= dictionary[:,:,0] + dictionary[:,:,1]*1j
    
    # remove any extra dimensions of size 1
    dictionary = np.squeeze(dictionary)
    lookup_table = np.squeeze(lookup_table)

    # load in the precomputed SVD vectors if they exist
    svd_path = data_folder+'/SVD.h5'
    if os.path.exists(svd_path):
        with h5py.File(svd_path, 'r') as f:
            U = f['U'][:]
            s = f['s'][:]
            Vh = f['Vh'][:]
    else:
        print("SVD file not found in path. Computing SVD...")
        # take L2-norm to normalise the data
        dictionary_norm = dictionary/np.linalg.norm(dictionary,axis=1, keepdims=True)
        U,s,Vh = np.linalg.svd(dictionary_norm, full_matrices=False) 
        print(f'Computed SVD. Saving to {svd_path}')
        with h5py.File(svd_path, 'w') as f:
            f.create_dataset('U', data = U, compression = 'gzip')
            f.create_dataset('s', data = s, compression = 'gzip') 
            f.create_dataset('Vh', data = Vh, compression = 'gzip') 
    
    return dictionary, lookup_table, U, s, Vh

def prepare_data(opt):
    """
    Prepares training sample for the networks, given path to training folder.
    
    Parameters:
    -----------
    opt : dict
        Dictionary of options for preparing data. Contains: 
            "svd" = number of svd components for subspace compression
            "labels" = parameter names (e.g. ["T1", "T2"])
            "foldername" = training data folder name, ontaining training data: dictionary (dictionary.h5) and SVD decomposition (SVD.h5).
    
    Returns:
    --------
    train_x : array [nSamps, nkComponents]
        Training features, normalised real-valued fingerprints
    train_y : array [nSamps, nParams]
        Scaled training labels (e.g. scaled T1/T2 values in range 0-1)
    nn : dict
        Normalisation and preprocessing parameters:
            scale: scaling factors for training labels
            V: low-rank subspace of dictionary
            P: phase-alignment angles
            normD: the actual norm of each fingerprint, normalised
    
    """

    # initialise network parameters
    nn = {}

    # load in training data
    D, lut, U, s, Vh = load_dic_svd(opt["foldername"])

    # normalise the input dictionary
    normD = np.sqrt(np.sum(np.abs(D)**2, axis=1))
    D = D / normD[:, np.newaxis] # normalise dictionary before SVD projection

    # scale the lookup table parameter range between 0 and 1
    label = lut[:,:] # scale all parameters
    nn['scale'] = np.max(label, axis = 0) # maximum scale factor across the label rows
    label = label/nn['scale'] 

    # low rank subspace of MRF dictionary
    V = Vh.conj().T
    V_k= V[:, :opt["svd"]] # reduce the components
    nn['V'] = V_k

    # project dictionary into low-rank space
    D_k = D @ V_k

    # Phase alignment of the compressed dictionary based on the first SVD-compressed component
    nn['P'] = np.angle(D_k[:, 0])
    D_k = D_k * np.exp(-1j * nn['P'][:, np.newaxis])
    D_k = np.real(D_k) # if dephasing is perfect then imag part is always zero    

    # Ensure dimension reduced dictionary is normalised
    tmp = np.sqrt(np.sum(np.abs(D_k)**2, axis=1))
    D_k = D_k / tmp[:, np.newaxis]

    nn['normD'] = normD * tmp

    train_x = D_k
    train_y = label
    
    return train_x, train_y, nn
    


   