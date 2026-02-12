"""
Utility functions for MRFResnet. 

 (c) 2025-2026: Ela Kanani, ela.kanani.21@ucl.ac.uk

"""
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import rcParams  

# Plotting settings for consistency
# Define the list of colours
colors = ['#ef476f', '#118ab2', '#f78c6b', '#ffd166', '#06d6a0', '#073b4c', '#008080']

# Set font family
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans']

# Set styling
mult = 2
style_params = {
    'label_fontsize': 12*mult,
    'tick_fontsize': 10*mult,
    'legend_title_fontsize': 12*mult,
    'legend_fontsize': 10*mult,
    'scatter_edgewidth': 1.05*mult,
    'axis_linewidth': 1.05*mult,
    'scatter_size': 25*mult,
    'legend_marker_size': 5*mult,
    'suplabel_font_size' : 14*mult
}



# --- 2. Set Global Matplotlib Parameters for Publication ---
plt.rcParams.update({
    'font.size': style_params['tick_fontsize'],
    'font.family': 'sans-serif',
    'font.sans-serif': 'Helvetica', # A standard Type 1 font family
    'svg.fonttype': 'none',         # For editable text in SVG (Inkscape)
    'ps.fonttype': 42,              # For embedding TrueType fonts in EPS/PS files
    'axes.linewidth': style_params['axis_linewidth']
})


def print_training_summary(model_training_info):
    """
        Print training statistics summary based on model.
    
    Parameters:
    -----------
        model_training_info : dict
            Model training history with keys: 'model', 'loss', 'epoch', 'lr'
    

    """

    print("\n" + "=" * 50)
    print(f"{model_training_info['model']} Training Summary.")
    print("=" * 50)

    epochs = model_training_info['epoch']
    train_loss = model_training_info['train_loss']
    val_loss = model_training_info['val_loss']
    
    
    print(f"Total epochs: {len(epochs)}")
    print(f"Initial Train loss: {train_loss[0]:.6f}")
    print(f"Final Train loss: {train_loss[-1]:.6f}")
    print(f"Best Train loss: {min(train_loss):.6f} (epoch {np.argmin(train_loss)+1:.0f})")
    print(f"Initial Validation loss: {val_loss[0]:.6f}")
    print(f"Final Validation loss: {val_loss[-1]:.6f}")
    print(f"Best Validation loss: {min(val_loss):.6f} (epoch {np.argmin(val_loss)+1:.0f})")


    return None


def plot_training_curves(model_training_info, model_ID, save_dir = './trained_models'):
    """
        Plot training curves (loss and learning rate) for model. Saves plots and displaus
    
    Parameters:
    -----------
        model_training_info : dict
            Model training history with keys: 'model', 'loss', 'epoch', 'lr'

        model_ID : str
        Timestamp/model ID for saving plots

        save_dir : str, optional
            Directory to save the plot 
    

    """
    model_type = model_training_info['model']

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f'{model_type} Training Curves', fontsize=style_params['suplabel_font_size'], fontweight='bold', y=1.00)

    axes[0].plot(model_training_info['epoch'], model_training_info['train_loss'], color = colors[0])
    axes[0].set_xlabel('Epoch', fontsize=style_params['label_fontsize'])
    axes[0].set_ylabel('Training Loss (MSE)', fontsize=style_params['label_fontsize'])
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_yscale('log')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    axes[1].plot(model_training_info['epoch'], model_training_info['val_loss'], color = colors[0])
    axes[1].set_xlabel('Epoch', fontsize=style_params['label_fontsize'])
    axes[1].set_ylabel('Validation Loss (MSE)', fontsize=style_params['label_fontsize'])
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_yscale('log')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)



    axes[2].plot(model_training_info['epoch'], model_training_info['lr'], color = colors[1])
    axes[2].set_xlabel('Epoch', fontsize=style_params['label_fontsize'])
    axes[2].set_ylabel('Learning Rate', fontsize=style_params['label_fontsize'])
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].set_yscale('log')
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)

    # Adjust layout and save
    plt.tight_layout()
    
    # Save plot
    plot_filename = f'{model_type}_{model_ID}_training_curves.png'
    plot_path = os.path.join(save_dir, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved: {plot_path}")
    
    # Display plot
    plt.show()