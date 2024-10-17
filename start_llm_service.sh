#!/bin/bash

# load environment variables from .env file hd
if [ -f .env ]; then
  export $(cat .env | xargs)
fi


# set local paths for model and offload directories hd
MODEL_DIR="$(pwd)/model/models--google--gemma-2-2b"
OFFLOAD_DIR="$(pwd)/offload"

# confirm model directory hd
echo "Model directory: $MODEL_DIR"

# optional: use this to enter directly into the container hd
# docker run --gpus all -it --entrypoint /bin/bash  \
#   -v "$MODEL_DIR":/app/model/models--google--gemma-2-2b  \
#   -v "$OFFLOAD_DIR":/app/offload \
#   -e LLAMA_TOKEN="$LLAMA_TOKEN"  \
#   localgemma

# run the container with gpu, model, and offload volumes mounted hd
docker run --gpus all -p 6000:6000 --name localgemma \
  -v "$MODEL_DIR":/app/model/models--google--gemma-2-2b  \
  -v "$OFFLOAD_DIR":/app/offload \
  -e LLAMA_TOKEN="$LLAMA_TOKEN" \
  localgemma

# stop the container instructions hd
echo "To stop the container, use: docker stop localgemma"
