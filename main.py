"""
main module for llm service

This script sets up a Flask-based API that loads a large language
model (LLM) using Hugging Face's transformers library. It includes
functions to load a pre-trained model and tokenizer from disk,
handle requests, and generate responses from the model.
"""

import os
import sys
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask, jsonify
from transformers import logging as hf_logging

# Configure Hugging Face transformers logging
hf_logging.set_verbosity(hf_logging.CRITICAL)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Suppress tokenizer logs from tokenization_utils_base
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# Get Hugging Face token
token = os.getenv('LLAMA_TOKEN')
if not token:
    logger.error("Hugging Face token (LLAMA_TOKEN) is not set in the environment.")
    sys.exit(1)

# Define model name and paths
model_name = "google/gemma-2-2b"
model_dir = "/app/model"
model_path = "/app/model/models--google--gemma-2-2b"
offload_folder = "/app/offload"

# Ensure model and offload directories exist
os.makedirs(model_dir, exist_ok=True)
os.makedirs(offload_folder, exist_ok=True)

# Initialize Flask app
app = Flask(__name__)

def get_model(model_path):
    """
    Load the model and tokenizer from the local directory if available.
    If not available, download it from Hugging Face.

    :param model_path: Path to the directory containing model files
    :type model_path: str
    :return: Loaded model and tokenizer or None if failed
    :rtype: tuple
    """
    try:
        logger.info(f"Looking for model in: {model_path}")

        # Check if model and tokenizer files exist
        if os.path.exists(os.path.join(model_path, 'model.safetensors.index.json')) and \
                os.path.exists(os.path.join(model_path, 'tokenizer_config.json')):
            logger.info("Loading model and tokenizer from cached safetensors files.")

            # Load tokenizer from local path
            # logging.disable(logging.CRITICAL)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            # logger.info(f"Tokenizer: {tokenizer}")

            # Load model with device map for inference
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",  # Automatically map model to devices
                offload_folder=offload_folder  # Offload to disk if needed
            )

            logger.info("Model loaded with device map and offloading.")
        else:
            # If model not found, download from Hugging Face
            logger.info("Model not found locally, downloading from Hugging Face...")

            tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                use_auth_token=token,
                device_map="auto",
                offload_folder=offload_folder
            )

            # Save the model and tokenizer locally for future use
            tokenizer.save_pretrained(model_path)
            model.save_pretrained(model_path)
            logger.info(f"Model and tokenizer downloaded and saved to {model_path}")

        return model, tokenizer

    except Exception as e:
        logger.exception("Error in get_model() function")
        return None, None

# Load model and tokenizer
model, tokenizer = get_model(model_path)

# Ensure the model is loaded before starting the server
if model is None or tokenizer is None:
    logger.error("Failed to load model or tokenizer. Exiting application.")
    sys.exit(1)

@app.route('/test/', methods=['GET'])
def test_llm():
    """
    Test endpoint to send a test query to the LLM and return the response

    :return: JSON response from the model
    :rtype: flask.Response
    """
    test_message = "Hi, are you awake?"

    try:
        # Prepare inputs
        inputs = tokenizer(test_message, return_tensors='pt')

        # Get device of model's parameters
        input_device = next(model.parameters()).device

        # Move inputs to the correct device
        inputs = {key: value.to(input_device) for key, value in inputs.items()}

        logger.info("Generating response from LLM...")

        # Generate output
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10)

        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return jsonify({'response': response_text})

    except Exception as e:
        logger.exception("Error during generation")
        return jsonify({'error': 'An internal error occurred'}), 500
