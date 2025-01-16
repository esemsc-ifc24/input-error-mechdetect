import os
import pandas as pd
import matplotlib.pyplot as plt

# Define paths
typo_dir = 'results_all-layers/text-sentences_typo/'
clean_dir = 'results_all-layers/text-sentences_clean/'
output_dir = 'analyses/plots/'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Layers to process
layers = [f'layer{i}.csv' for i in range(13)]

# Number of rows to process (excluding the header)
num_rows = 3

# Number of neurons per plot
neurons_per_plot = 200

# Iterate through each layer file
for layer_file in layers:
    typo_path = os.path.join(typo_dir, layer_file)
    clean_path = os.path.join(clean_dir, layer_file)

    # Load typo and clean activations, skipping the header
    typo_activations = pd.read_csv(typo_path, header=0).iloc[:num_rows]
    clean_activations = pd.read_csv(clean_path, header=0).iloc[:num_rows]

    # Total number of neurons
    total_neurons = typo_activations.shape[1]

    # Create sub-plots for each range of neurons
    for start in range(0, total_neurons, neurons_per_plot):
        end = start + neurons_per_plot

        # Create plot
        plt.figure(figsize=(10, 6))

        # Plot typo activations in shades of blue
        for i, row in enumerate(typo_activations.iterrows()):
            alpha = (i + 1) / num_rows  # Gradually increase opacity
            plt.plot(row[1].values[start:end], color=(0.2, 0.4, 1, alpha), label='Typo' if i == 0 else "")

        # Plot clean activations in shades of red
        for i, row in enumerate(clean_activations.iterrows()):
            alpha = (i + 1) / num_rows  # Gradually increase opacity
            plt.plot(row[1].values[start:end], color=(1, 0.2, 0.2, alpha), label='Clean' if i == 0 else "")

        # Add labels and legend
        plt.title(f'Layer {layer_file.split('.')[0]} Activations (Neurons {start}-{end})')
        plt.xlabel('Neuron Index')
        plt.ylabel('Activation Value')
        plt.legend()

        # Save plot
        output_path = os.path.join(output_dir, f'{layer_file.split('.')[0]}_activations_{start}_{end}.png')
        plt.savefig(output_path)
        plt.close()

print(f"Plots saved in {output_dir}")
