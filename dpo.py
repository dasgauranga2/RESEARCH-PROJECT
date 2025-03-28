import json
import os
from urllib import request
import pprint
import torch
from torch.utils.data import Dataset
#import tiktoken
from torch.utils.data import DataLoader
from functools import partial
#from util import GPTModel
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

########################################################################################
# IMPORTANT
# By default, this may run on multiple GPUs which could cause problems
# Therefore, run the script using the below command which runs only on CUDA device 0
# CUDA_VISIBLE_DEVICES=0 python dpo.py
########################################################################################

SEED = 10

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# data is a list of dictionaries
with open("data/ultrafeedback.json", "r", encoding="utf-8") as file:
    data = json.load(file)
print("Number of entries:", len(data))

# each example is a dictionary containing keys
# 'instruction' and 'inputs' which are part of the prompt and
# 'chosen' and 'rejected' which represents the responses
#pprint.pp(data[0])

START_TEXT = '<|begin_of_text|>'
END_TEXT = '<|end_of_text|>'
START_HEADER = '<|start_header_id|>'
END_HEADER = '<|end_header_id|>'
PADDING_TOKEN = '<|finetune_right_pad_id|>'
END_TURN = '<|eot_id|>'

# function that combines the 'instruction' and 'inputs'
# in a single text to form the prompt and formats it according to the 
# model's expected input template
# this input template is for gpt-2
# change it for other models
def format_input(entry):
    # instruction_text = (
    #     f"Below is an instruction that describes a task. "
    #     f"Write a response that appropriately completes the request."
    #     f"\n\n### Instruction:\n{entry['instruction']}"
    # )

    input_text = START_TEXT + START_HEADER + 'user' + END_HEADER + '\n\n' + entry['instruction'] + END_TURN

    return input_text

#print(format_input(data[0]))

# divide the datasets into training and validation sets
train_size = int(len(data) * 0.85)

train_data = data[:train_size]
val_data = data[train_size:]

#print(len(train_data),len(val_data))

# preference dataset class
class PreferenceDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # list that will contain input tokens
        self.encoded_texts = []

        # iterate over the original data
        for entry in data:
            # create the prompt text from the 'instruction' and 'input'
            prompt = format_input(entry)
            # get the rejected response
            rejected_response = entry["rejected"]
            # get the chosen response
            chosen_response = entry["chosen"]
            # get the label
            label = entry["label"]

            # tokenize the prompt
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            # prompt + chosen response text
            chosen_full_text = f"{prompt}\n{START_HEADER}assistant{END_HEADER}\n{chosen_response}{END_TURN}"
            # prompt + rejected response text
            rejected_full_text = f"{prompt}\n{START_HEADER}assistant{END_HEADER}\n{rejected_response}{END_TURN}"
            # tokenize the prompt + chosen response text
            chosen_full_tokens = tokenizer.encode(chosen_full_text, add_special_tokens=False)
            # tokenize the prompt + rejected response text
            rejected_full_tokens = tokenizer.encode(rejected_full_text, add_special_tokens=False)

            self.encoded_texts.append({
                "prompt": prompt_tokens, # prompt text token ids
                "chosen": chosen_full_tokens,  # prompt + chosen response text token ids
                "rejected": rejected_full_tokens,  # prompt + rejected response text token ids
                "label": label
            })

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)
    
# function to decode the token ids to text to verify the dataset
def decode_tokens_from_list(token_ids, tokenizer):
    return list(tokenizer.convert_ids_to_tokens(token_ids))

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
#print(tokenizer.eos_token)

example_dataset = PreferenceDataset(data[:4], tokenizer)
#print(example_dataset[0])
#print(decode_tokens_from_list(example_dataset[0]['prompt'],tokenizer))
#print(decode_tokens_from_list(example_dataset[0]['chosen'],tokenizer))
#print(decode_tokens_from_list(example_dataset[0]['rejected'],tokenizer))
#print(example_dataset[0]['label'])
    
# batch collator function to create a batch of inputs
def custom_collate_fn(
    batch,
    pad_token_id=tokenizer.pad_token_id,
    allowed_max_length=None,
    mask_prompt_tokens=True,
    device="cpu"
):
    # batch above is a list of data points
    # each data point is a dictionary containing
    # prompt text token ids, prompt + chosen response text token ids, prompt + rejected response text token ids

    # Initialize lists to hold batch data
    batch_data = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
        "rejected_mask": [],
        "chosen_mask": [],
        "label": []
    }

    # Determine the longest sequence to set a common padding length
    max_length_common = 0
    if batch:
        for key in ["chosen", "rejected"]:
            current_max = max(len(item[key])+1 for item in batch)
            max_length_common = max(max_length_common, current_max)

    # Process each item in the batch
    for item in batch:
        prompt = torch.tensor(item["prompt"])
        batch_data["prompt"].append(prompt)

        label = torch.tensor(item["label"])
        batch_data["label"].append(label)

        for key in ["chosen", "rejected"]:
            # Adjust padding according to the common maximum length
            sequence = item[key]
            padded = sequence + [pad_token_id] * (max_length_common - len(sequence))
            mask = torch.ones(len(padded)).bool()

            # Set mask for all padding tokens to False
            mask[len(sequence):] = False

            # Set mask for all prompt tokens to False
            # +1 sets the newline ("\n") token to False
            if mask_prompt_tokens:
                mask[:prompt.shape[0]+1] = False

            batch_data[key].append(torch.tensor(padded))
            batch_data[f"{key}_mask"].append(mask)

    # Final processing
    for key in ["chosen", "rejected", "chosen_mask", "rejected_mask"]:
        # Stack all sequences into a tensor for the given key
        tensor_stack = torch.stack(batch_data[key])

        # Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            tensor_stack = tensor_stack[:, :allowed_max_length]

        # Move to the specified device
        batch_data[key] = tensor_stack.to(device)

    batch_data["label"] = torch.tensor(batch_data["label"])

    return batch_data

device = torch.device("cuda")

num_workers = 0
batch_size = 6

customized_collate_fn = partial(
    custom_collate_fn,
    device=device,            # Put the data directly on a GPU if available
    mask_prompt_tokens=True,  # This is optional
    allowed_max_length=1024   # The supported context length of the model
)

train_dataset = PreferenceDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

val_dataset = PreferenceDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

# function to filter a batch and
# select only those samples that have a certain label value
def filter_batch_by_label(batch, target_label=1):
    
    mask = batch["label"] == target_label
    filtered_batch = {}

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            filtered_batch[key] = value[mask]
        else:
            # For lists (like list of tensors), convert to tensor first if needed
            filtered_batch[key] = [v for i, v in enumerate(value) if mask[i]]

    return filtered_batch

# sampling a batch to verify the data
# batch = next(iter(train_loader))
# print(batch)
# print(filter_batch_by_label(batch), 1)
# # this should only contain the prompt text tokens
# print(decode_tokens_from_list(batch['prompt'][0],tokenizer))
# # this should contain prompt + chosen response text + <|endoftext|> tokens (<|endoftext|> are padding tokens)
# print(decode_tokens_from_list(batch['chosen'][0],tokenizer))
# # this should only contain the chosen response text tokens (exlcudes the prompt tokens and padding tokens)
# print(decode_tokens_from_list(batch['chosen'][0][batch["chosen_mask"][0]],tokenizer))
# # this should contain prompt + rejected response text + <|endoftext|> tokens (<|endoftext|> are padding tokens)
# print(decode_tokens_from_list(batch['rejected'][0],tokenizer))
# # this should only contain the rejected response text tokens (exlcudes the prompt tokens and padding tokens)
# print(decode_tokens_from_list(batch['rejected'][0][batch["rejected_mask"][0]],tokenizer))
# print(batch['label'])

# load the language model to be trained
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
policy_model = model
# load the reference model
reference_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
reference_model.eval()

policy_model.to(device)
reference_model.to(device)

# function to calculate the DPO loss given log-probabilities
# of chosen and rejected response from the language and reference model
def compute_dpo_loss(
      model_chosen_logprobs,
      model_rejected_logprobs,
      reference_chosen_logprobs,
      reference_rejected_logprobs,
      beta=0.1,
    ):
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logprobs: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logprobs: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logprobs: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logprobs: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss; typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss.

    Returns:
        A tuple of three tensors: (loss, chosen_rewards, rejected_rewards).
    """

    model_logratios = model_chosen_logprobs - model_rejected_logprobs
    reference_logratios = reference_chosen_logprobs - reference_rejected_logprobs
    logits = model_logratios - reference_logratios

    # calculate the final loss
    losses = -F.logsigmoid(beta * logits)

    # calculate the implicit reward by the model for the chosen responses
    chosen_rewards = (model_chosen_logprobs - reference_chosen_logprobs).detach()
     # calculate the implicit reward by the model for the rejected responses
    rejected_rewards = (model_rejected_logprobs - reference_rejected_logprobs).detach()

    # calculate the average over the samples in the batch
    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()

# function to calculate the log-probabilities
# given the model outputs, labels and mask
# this calculates log(Pr(y|x))
def compute_logprobs(logits, labels, selection_mask=None):
    """
    Args:
      logits: Tensor of shape (batch_size, num_tokens, vocab_size)
      labels: Tensor of shape (batch_size, num_tokens)
      selection_mask: Tensor for shape (batch_size, num_tokens)

    Returns:
      mean_log_prob: Mean log probability excluding padding tokens.
    """
    #print(logits.shape)
    
    # labels are the inputs shifted by one
    labels = labels[:, 1:].clone()

    # truncate logits to match the labels num_tokens
    logits = logits[:, :-1, :]

    # calculate the log-probabilities from the model outputs
    log_probs = F.log_softmax(logits, dim=-1)

    # gather the log probabilities for the actual labels (next token)
    # each now contains log-probability of token of next-time step given
    # the current and previous time-steps
    selected_log_probs = torch.gather(
        input=log_probs,
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)

    if selection_mask is not None:
        # shift the mask by one
        mask = selection_mask[:, 1:].clone()

        # apply the mask to only select log-probabilities of response tokens
        # tokens corresponding to prompt and padding are removed
        selected_log_probs = selected_log_probs * mask

        # finally, log-probability of the response given the prompt is calculated
        # as the average log-probability of all the response tokens
        avg_log_prob = selected_log_probs.sum(-1) / mask.sum(-1)

        return avg_log_prob

    else:
        return selected_log_probs.mean(-1)

# function to calculate the DPO loss of an entire batch of data
def compute_dpo_loss_batch(batch, policy_model, reference_model, beta):

    # log-probabilities of y_w given x from the language model
    policy_chosen_log_probas = compute_logprobs(
        logits=policy_model(batch["chosen"]).logits,
        labels=batch["chosen"],
        selection_mask=batch["chosen_mask"]
    )
    # log-probabilities of y_l given x from the language model
    policy_rejected_log_probas = compute_logprobs(
        logits=policy_model(batch["rejected"]).logits,
        labels=batch["rejected"],
        selection_mask=batch["rejected_mask"]
    )
    
    with torch.no_grad():
        # log-probabilities of y_w given x from the reference model
        ref_chosen_log_probas = compute_logprobs(
            logits=reference_model(batch["chosen"]).logits,
            labels=batch["chosen"],
            selection_mask=batch["chosen_mask"]
        )
        # log-probabilities of y_l given x from the reference model
        ref_rejected_log_probas = compute_logprobs(
            logits=reference_model(batch["rejected"]).logits,
            labels=batch["rejected"],
            selection_mask=batch["rejected_mask"]
        )

    # compute the final loss
    loss, chosen_rewards, rejected_rewards = compute_dpo_loss(
        model_chosen_logprobs=policy_chosen_log_probas,
        model_rejected_logprobs=policy_rejected_log_probas,
        reference_chosen_logprobs=ref_chosen_log_probas,
        reference_rejected_logprobs=ref_rejected_log_probas,
        beta=beta
    )
    return loss, chosen_rewards, rejected_rewards

# function to evaluate the model on an entire dataset
def compute_dpo_loss_loader(data_loader, policy_model, reference_model, beta, num_batches=None):

    total_loss, total_chosen_rewards, total_rejected_rewards = 0., 0., 0.
    if len(data_loader) == 0:
        return float("nan")

    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))

    filtered_batch_count = 0

    for i, batch in enumerate(data_loader):
        # get only those samples from the batch that have label = 1
        batch = filter_batch_by_label(batch, target_label=1)

        # skip empty filtered batch
        if batch["label"].numel() == 0:
            continue

        if i < num_batches:
            loss, chosen_rewards, rejected_rewards = compute_dpo_loss_batch(
                batch=batch,
                policy_model=policy_model,
                reference_model=reference_model,
                beta=beta
            )
            total_loss += loss.item()
            total_chosen_rewards += chosen_rewards.item()
            total_rejected_rewards += rejected_rewards.item()
            filtered_batch_count += 1

        else:
            break

    if filtered_batch_count == 0:
        return float("nan")

    # calculate average
    total_loss /= filtered_batch_count
    total_chosen_rewards /= filtered_batch_count
    total_rejected_rewards /= filtered_batch_count
    return total_loss, total_chosen_rewards, total_rejected_rewards

# function to evaluate the model on both the training and validation sets
def evaluate_dpo_loss_loader(policy_model, reference_model, train_loader, val_loader, beta, eval_iter):

    policy_model.eval()
    with torch.no_grad():
        train_loss, train_chosen_rewards, train_rejected_rewards = compute_dpo_loss_loader(
            data_loader=train_loader,
            policy_model=policy_model,
            reference_model=reference_model,
            beta=beta,
            num_batches=eval_iter
        )

        val_loss, val_chosen_rewards, val_rejected_rewards = compute_dpo_loss_loader(
            data_loader=val_loader,
            policy_model=policy_model,
            reference_model=reference_model,
            beta=beta,
            num_batches=eval_iter
        )

    res = {
        "train_loss": train_loss,
        "train_chosen_reward": train_chosen_rewards,
        "train_rejected_reward": train_rejected_rewards,
        "val_loss": val_loss,
        "val_chosen_reward": val_chosen_rewards,
        "val_rejected_reward": val_rejected_rewards
    }

    policy_model.train()
    return res

# optimizer
optimizer = torch.optim.AdamW(policy_model.parameters(), lr=5e-6, weight_decay=0.01)

EPOCHS = 6
BETA = 0.1
steps = -1
# frequency at which the model will be evaluated
eval_freq = 50
# when the model is evaluated this variable
# decides the number of batches that will be used to calculate the metrics
eval_iter = 20

for epoch in range(EPOCHS):
    # set the language model to training mode
    policy_model.train()

    for batch_idx, batch in enumerate(train_loader):

        optimizer.zero_grad()

        # use the batch of data to compute the loss
        loss, chosen_rewards, rejected_rewards = compute_dpo_loss_batch(
            batch=batch,
            policy_model=policy_model,
            reference_model=reference_model,
            beta=BETA
        )

        # calculate the gradients of the loss
        loss.backward()
        # update the model weights 
        optimizer.step()

        steps += 1

        if steps % eval_freq == 0:
            # evaluate the model
            res = evaluate_dpo_loss_loader(
                policy_model=policy_model,
                reference_model=reference_model,
                train_loader=train_loader,
                val_loader=val_loader,
                beta=BETA,
                eval_iter=eval_iter
            )

            train_reward_margin = res["train_chosen_reward"] - res["train_rejected_reward"]
            val_reward_margin = res["val_chosen_reward"] - res["val_rejected_reward"]

            print(
                f"Ep {epoch+1} (Step {steps:03d}): "
                f"Train loss {res['train_loss']:.3f}, Val loss {res['val_loss']:.3f}, "
                f"Train reward margins {train_reward_margin:.3f}, "
                f"Val reward margins {val_reward_margin:.3f}"
            )