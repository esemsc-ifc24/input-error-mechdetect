import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define paths
typo_dir = 'results_all-layers/text-sentences_typo/'
clean_dir = 'results_all-layers/text-sentences_clean/'
output_dir = 'analyses/plots/3-layers'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Number of lines to plot
num_lines = 3

# Iterate through layers
for layer_file in sorted(os.listdir(typo_dir)):
    if layer_file.endswith('.csv'):
        layer_name = layer_file.split('.')[0]

        # Load typo and clean activations
        typo_path = os.path.join(typo_dir, layer_file)
        clean_path = os.path.join(clean_dir, layer_file)

        typo_activations = pd.read_csv(typo_path, header=None).iloc[:num_lines]
        clean_activations = pd.read_csv(clean_path, header=None).iloc[:num_lines]

        # Start plotting
        plt.figure(figsize=(10, 6))

        # Plot typo activations in shades of blue
        for i, row in enumerate(typo_activations.iterrows()):
            alpha = (i + 1) / num_lines  # Gradually increase opacity
            plt.plot(row[1].values, color=(0.2, 0.4, 1, alpha), label='Typo' if i == 0 else "")

        # Plot clean activations in shades of orange
        for i, row in enumerate(clean_activations.iterrows()):
            alpha = (i + 1) / num_lines  # Gradually increase opacity
            plt.plot(row[1].values, color=(1, 0.5, 0, alpha), label='Clean' if i == 0 else "")

        # Add labels and legend
        plt.title(f'Layer {layer_name} Activations')
        plt.xlabel('Neuron Index')
        plt.ylabel('Activation Value')
        plt.legend()

        # Save plot
        output_path = os.path.join(output_dir, f'{layer_name}_activations.png')
        plt.savefig(output_path)
        plt.close()

print(f"Plots saved in {output_dir}")
