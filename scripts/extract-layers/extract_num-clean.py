import os
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from datetime import datetime
import traceback

# Global variable for the input file
INPUT_FILE = "number-sentences_clean.csv"

# Paths
PROCESSED_DATA_PATH = "data/processed/"
RESULTS_PATH = "results_layers/"
LOG_FILE = "logs/extract_activations_layers.log"

# Model configuration
MODEL_NAME = "bert-base-uncased"

# Load tokenizer and model
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True)
    model.eval()
except Exception as e:
    with open(LOG_FILE, "a") as log_file:
        log_file.write(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] " +
            f"Model loading failed: {str(e)}\n")
    raise e

# Layers to extract
LAYERS_TO_EXTRACT = [1, 4, 8, 12]


# Function to log messages
def log(message):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, "a") as log_file:
            log_file.write(f"[{timestamp}] {message}\n")
        print(message)
    except Exception as e:
        print(f"Logging failed: {str(e)}")


# Function to extract activations from specified layers
def extract_activations(file_name, run_id):
    try:
        log(f"Processing file: {file_name}, Run: {run_id}")
        file_path = os.path.join(PROCESSED_DATA_PATH, file_name)
        df = pd.read_csv(
            file_path, header=None, names=['sentence', 'tokenized'])

        for layer in LAYERS_TO_EXTRACT:
            activations = []
            for sentence in df['sentence']:
                inputs = tokenizer(sentence, return_tensors="pt",
                                   truncation=True,
                                   padding="max_length", max_length=128)
                with torch.no_grad():
                    outputs = model(**inputs)
                hidden_state = outputs.hidden_states[layer].mean(
                    dim=1).squeeze().tolist()
                activations.append(hidden_state)

            # Save results
            output_folder = os.path.join(RESULTS_PATH, file_name.split('.')[0])
            os.makedirs(output_folder, exist_ok=True)
            result_file = os.path.join(
                output_folder, f"run{run_id}_layer{layer}.csv")
            pd.DataFrame(activations).to_csv(result_file, index=False)
            log(f"Activations saved: {result_file}")

    except Exception as e:
        log(f"Error processing file {file_name}, Run {run_id}: {str(e)}")
        log(traceback.format_exc())


# Main
try:
    os.makedirs(RESULTS_PATH, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    log("Starting layer-wise activation extraction...")

    # Determine number of runs based on file type
    num_runs = 5 if INPUT_FILE.startswith("number-sentences_") else 3

    for run_id in range(1, num_runs + 1):
        extract_activations(INPUT_FILE, run_id)

    log("Layer-wise activation extraction completed.")
except Exception as e:
    log(f"Unexpected error: {str(e)}")
    log(traceback.format_exc())
