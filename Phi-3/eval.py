import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)

# load the model
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct", 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True, 
    attn_implementation="flash_attention_2"
)

# load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")

# conversation in list format
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
    {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
    {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
]

# create the input token ids
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to('cuda')
print(inputs)

# get the model outputs
outputs = model.generate(inputs, max_new_tokens=500)
# get the generated text
text = tokenizer.batch_decode(outputs)[0]
print(text)