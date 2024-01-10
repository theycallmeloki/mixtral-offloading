#!/bin/bash

# Download the model
huggingface-cli download lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo --quiet --local-dir Mixtral-8x7B-Instruct-v0.1-offloading-demo

# Execute the command provided to the docker run command
exec "$@"
