import sys

sys.path.append("mixtral-offloading")
import torch
from torch.nn import functional as F
from hqq.core.quantize import BaseQuantizeConfig
from huggingface_hub import snapshot_download
# from IPython.display import clear_output
from flask import Flask, request, Response
from tqdm.auto import trange
from transformers import AutoConfig, AutoTokenizer
from transformers.utils import logging as hf_logging

from src.build_model import OffloadConfig, QuantConfig, build_model


# Function to get total GPU memory in GB
def get_gpu_memory():
    total_memory = torch.cuda.get_device_properties(0).total_memory
    return total_memory / (1024 ** 3)  # Convert bytes to GB


# Function to determine offload_per_layer based on GPU memory
def determine_offload_per_layer():
    gpu_memory_gb = get_gpu_memory()
    offload_value = int(gpu_memory_gb / 4)
    # Ensure the value is within 1 to 5
    return max(1, min(offload_value, 5))

# Set the offload_per_layer value
offload_per_layer = determine_offload_per_layer()

print(f"GPU Memory: {get_gpu_memory()} GB")
print(f"Set offload_per_layer to: {offload_per_layer}")

# Initialize Flask app
app = Flask(__name__)

model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
quantized_model_name = "lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo"
state_path = "Mixtral-8x7B-Instruct-v0.1-offloading-demo"

config = AutoConfig.from_pretrained(quantized_model_name)

device = torch.device("cuda:0")

num_experts = config.num_local_experts

offload_config = OffloadConfig(
    main_size=config.num_hidden_layers * (num_experts - offload_per_layer),
    offload_size=config.num_hidden_layers * offload_per_layer,
    buffer_size=4,
    offload_per_layer=offload_per_layer,
)


attn_config = BaseQuantizeConfig(
    nbits=4,
    group_size=64,
    quant_zero=True,
    quant_scale=True,
)
attn_config["scale_quant_params"]["group_size"] = 256


ffn_config = BaseQuantizeConfig(
    nbits=2,
    group_size=16,
    quant_zero=True,
    quant_scale=True,
)
quant_config = QuantConfig(ffn_config=ffn_config, attn_config=attn_config)


model = build_model(
    device=device,
    quant_config=quant_config,
    offload_config=offload_config,
    state_path=state_path,
)

from transformers import TextStreamer


tokenizer = AutoTokenizer.from_pretrained(model_name)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

@app.route('/generate', methods=['POST'])
def generate():
    # Extract data from request outside the generator function
    data = request.json
    user_input = data.get('input')
    user_entry = dict(role="user", content=user_input)
    input_ids = tokenizer.apply_chat_template([user_entry], return_tensors="pt").to(torch.device("cuda:0"))

    def generate_stream():
        # Now use the pre-extracted data
        # Generate response using Mixtral
        result = model.generate(
            input_ids=input_ids,
            streamer=streamer,
            do_sample=True,
            temperature=0.9,
            top_p=0.9,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )
        
        # Stream the response
        sequence = result["sequences"]
        yield tokenizer.decode(sequence[0], skip_special_tokens=True)

    return Response(generate_stream(), mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)