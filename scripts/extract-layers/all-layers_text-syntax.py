import os
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from datetime import datetime
import traceback

# Global variable for the input file
INPUT_FILE = "text-sentences_syntax-error.csv"

# Paths
PROCESSED_DATA_PATH = "data/processed/"
RESULTS_PATH = "results_all-layers/"
LOG_FILE = "logs/all-layers_text-syntax.log"

# Model configuration
MODEL_NAME = "bert-base-uncased"

# Load tokenizer and model
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True)
    model.eval()
except Exception as e:
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] " + 
                       "Model loading failed: {str(e)}\n")
    raise e


# Function to log messages
def log(message):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, "a") as log_file:
            log_file.write(f"[{timestamp}] {message}\n")
        print(message)
    except Exception as e:
        print(f"Logging failed: {str(e)}")


# Function to extract activations from all layers
def extract_activations(file_name):
    try:
        log(f"Processing file: {file_name}")
        file_path = os.path.join(PROCESSED_DATA_PATH, file_name)
        df = pd.read_csv(
            file_path, header=None, names=['sentence', 'tokenized'])

        # Initialize storage for layer activations
        layer_activations = {layer: [] for layer in range(model.config.num_hidden_layers + 1)}

        for sentence in df['sentence']:
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
                for layer in range(len(outputs.hidden_states)):
                    hidden_state = outputs.hidden_states[layer].mean(dim=1).squeeze().tolist()
                    layer_activations[layer].append(hidden_state)

        # Save activations for each layer
        output_folder = os.path.join(RESULTS_PATH, file_name.split('.')[0])
        os.makedirs(output_folder, exist_ok=True)
        for layer, activations in layer_activations.items():
            result_file = os.path.join(output_folder, f"layer{layer}.csv")
            pd.DataFrame(activations).to_csv(result_file, index=False)
            log(f"Activations for layer {layer} saved: {result_file}")

    except Exception as e:
        log(f"Error processing file {file_name}: {str(e)}")
        log(traceback.format_exc())


# Main
try:
    os.makedirs(RESULTS_PATH, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    log("Starting full-layer activation extraction...")
    extract_activations(INPUT_FILE)
    log("Activation extraction completed.")
except Exception as e:
    log(f"Unexpected error: {str(e)}")
    log(traceback.format_exc())
