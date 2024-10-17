# Llama model 3.2-3b

This is a local LLM (Large Language Model) service running in a Docker container. It uses the Llama 3.23b model and anticipates CUDA-compatible hardware for optimal performance. The application leverages Gunicorn to provide an easy-to-use API for generating text based on prompts, and it's packaged for simple deployment.

## Project Structure

```
my_docker_project/
|— Dockerfile
|— app.py
|— requirements.txt
|— README.md
|— .dockerignore
```

## Setup and Running Instructions

### Prerequisites
- Docker installed on your system. You can find installation instructions [here](https://docs.docker.com/get-docker/).
- NVIDIA drivers and CUDA installed for GPU support. Refer to the [NVIDIA CUDA Toolkit Documentation](https://developer.nvidia.com/cuda-toolkit) for more information.
- Two local directories one for offloading model sections durreing inference and one for storing the trained model
- A .env file with a token for the required model from hugginface

### Build the Docker Image
To build the Docker image, run the following command from the root of the project directory:

```sh
docker build -t localgemma .
```

The Dockerfile will try to build using the NVIDIA runtime PyTorch image first. If it fails, it will fall back to a base CUDA image and create the environment from scratch.

### Run the Docker Container

To run the Docker container, use the following command:

```sh
docker run --gpus all -p 6000:6000 --name localgemma \
  -v "$MODEL_DIR":/app/model/models--google--gemma-2-2b  \
  -v "$OFFLOAD_DIR":/app/offload \
  -e LLAMA_TOKEN="$LLAMA_TOKEN" \
  localgemma
```

The Flask application will be accessible at `http://localhost:5440`.

### Endpoints

- `/test` (GET): verifies the llm responds to prompt
-  more coming soon

### Notes

- Make sure your machine has a compatible GPU and drivers if you want to take advantage of CUDA for model inference.
- Cuda and the toolkit take some time to impliment for eacb machine
- Update the `model_path` in `app.py` to point to the correct model location or adjust to use a model available from Hugging Face's hub.

### Additional Resources
- [Docker](https://www.docker.com/): Container platform for building and running applications.
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit): Toolkit for developing GPU-accelerated applications.
- [Hugging Face](https://huggingface.co/): Platform for sharing machine learning models and datasets.
- [LangChain](https://langchain.com/): Framework for building applications with LLMs.
- [Ollama](https://ollama.com/): A solution for deploying LLMs locally or in cloud environments.

These resources provide further information on the technologies used in this project and can help with expanding the current implementation.
