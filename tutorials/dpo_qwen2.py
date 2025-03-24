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
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from datasets import load_dataset
from qwen_vl_utils import process_vision_info

########################################################################################
# IMPORTANT
# By default, this may run on multiple GPUs which could cause problems
# Therefore, run the script using the below command which runs only on CUDA device 0
# CUDA_VISIBLE_DEVICES=0 python dpo_qwen2.py
########################################################################################

# set seed for reproducibility
SEED = 10

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# load the dataset
data = load_dataset("openbmb/RLAIF-V-Dataset", split='train')
# select only the first 100 data points
data = data.select(range(100))
print("Number of entries:", len(data))

# each example is a dictionary containing keys
# 'question', 'chosen', 'rejected' and a few more keys
pprint.pp(data[0])

# training set size
train_size = int(len(data) * 0.85)

# divide the datasets into training and validation sets
train_data = data.select(range(0,train_size))
val_data = data.select(range(train_size,len(data)))

# these parameters refer to the total number of pixels (i.e., width × height) an image should have after resizing
# this can be used to control the no. of image tokens in the input sequence
# min_pixels is the minimum number of pixels allowed after resizing
# max_pixels is the maximum number of pixels allowed
min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", 
                                          min_pixels=min_pixels, 
                                          max_pixels=max_pixels,
                                          use_fast=True)

#example_dataset = PreferenceDataset(data[:4], tokenizer)
#print(example_dataset[0])
#print(decode_tokens_from_list(example_dataset[0]['prompt'],tokenizer))
#print(decode_tokens_from_list(example_dataset[0]['chosen'],tokenizer))
#print(decode_tokens_from_list(example_dataset[0]['rejected'],tokenizer))
    
# batch collator function to create a batch of inputs
def custom_collate_fn(
    batch,
    allowed_max_length=None,
    mask_prompt_tokens=True,
    device="cpu"
):
    # batch above is a list of data points
    # each data point is a dataset example

    # list of prompt + chosen response texts
    chosen_messages = []
    # list of prompt + rejected response texts
    rejected_messages = []

    for item in batch:
        
        # each input sequence should be in this format

        chosen_message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": item['image'],
                        # resize the image to adjust no. of image tokens
                        "resized_height": 280,
                        "resized_width": 280,
                    },
                    {"type": "text", "text": item['question']},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": item['chosen']},
                ],
            }
        ]

        rejected_message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": item['image'],
                        # resize the image to adjust no. of image tokens
                        "resized_height": 280,
                        "resized_width": 280,
                    },
                    {"type": "text", "text": item['question']},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": item['rejected']},
                ],
            }
        ]

        chosen_messages.append(chosen_message)
        rejected_messages.append(rejected_message)

    # use processor to apply the chat template
    # and format the texts
    chosen_texts = [
        processor.apply_chat_template(msg, tokenize=False)
        for msg in chosen_messages
    ]

    rejected_texts = [
        processor.apply_chat_template(msg, tokenize=False)
        for msg in rejected_messages
    ]

    # get the vision inputs
    chosen_image_inputs, chosen_video_inputs = process_vision_info(chosen_messages)
    rejected_image_inputs, rejected_video_inputs = process_vision_info(rejected_messages)

    # create the inputs for the model
    chosen_inputs = processor(
        text=chosen_texts,
        images=chosen_image_inputs,
        videos=chosen_video_inputs,
        padding=True,
        return_tensors="pt",
    )
    rejected_inputs = processor(
        text=rejected_texts,
        images=rejected_image_inputs,
        videos=rejected_video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    chosen_inputs = chosen_inputs.to(device)
    rejected_inputs = rejected_inputs.to(device)

    # start token id
    start_token_id = processor.tokenizer.convert_tokens_to_ids('<|im_start|>')
    # assistant token id
    assistant_token_id = processor.tokenizer.convert_tokens_to_ids('assistant')

    # creating the masks
    # the masks will ensure that loss is calculated
    # only for the response tokens
    chosen_mask = torch.zeros_like(chosen_inputs['input_ids'])
    for i in range(len(chosen_mask)):
        for j in range(len(chosen_mask[i])):
            if chosen_inputs['input_ids'][i][j]==start_token_id and chosen_inputs['input_ids'][i][j+1]==assistant_token_id:
                chosen_mask[i,j:] = 1

    rejected_mask = torch.zeros_like(rejected_inputs['input_ids'])
    for i in range(len(rejected_mask)):
        for j in range(len(rejected_mask[i])):
            if rejected_inputs['input_ids'][i][j]==start_token_id and rejected_inputs['input_ids'][i][j+1]==assistant_token_id:
                rejected_mask[i,j:] = 1

    batch_data = {
        #"prompt": [],
        "chosen": chosen_inputs,
        "rejected": rejected_inputs,
        "chosen_mask": chosen_mask.bool(),
        "rejected_mask": rejected_mask.bool()
    }
    #print(batch_data)

    return batch_data

device = torch.device("cuda")

num_workers = 0
# batch size
batch_size = 2

# create a funciton with some of it's parameters filled
customized_collate_fn = partial(
    custom_collate_fn,
    device=device,            # Put the data directly on a GPU if available
    mask_prompt_tokens=True,  # This is optional
    allowed_max_length=1024   # The supported context length of the model
)

# create the data loaders
train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

val_loader = DataLoader(
    val_data,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

# function to decode the token ids to text to verify the dataset
def decode_tokens_from_list(token_ids, processor):
    return processor.tokenizer.convert_ids_to_tokens(token_ids)

# # sampling a batch to verify the data
# batch = next(iter(example_loader))
# # # this should only contain the prompt text tokens
# # print(decode_tokens_from_list(batch['prompt'][0],tokenizer))
# # this should contain prompt + chosen response text + <|endoftext|> tokens (<|endoftext|> are padding tokens)
# for i in range(batch_size):
#     print(decode_tokens_from_list(batch['rejected']['input_ids'][i],processor))
#     print(decode_tokens_from_list(batch['rejected']['input_ids'][i][batch['rejected_mask'][i]],processor))
# # this should only contain the chosen response text tokens (exlcudes the prompt tokens and padding tokens)
# print(decode_tokens_from_list(batch['chosen'][0][batch["chosen_mask"][0]],tokenizer))
# # this should contain prompt + rejected response text + <|endoftext|> tokens (<|endoftext|> are padding tokens)
# print(decode_tokens_from_list(batch['rejected'][0],tokenizer))
# # this should only contain the rejected response text tokens (exlcudes the prompt tokens and padding tokens)
# print(decode_tokens_from_list(batch['rejected'][0][batch["rejected_mask"][0]],tokenizer))

# function to create the model
def create_model():
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    return model

# load the language model to be trained
model = create_model()
policy_model = model
# load the reference model
reference_model = create_model()
reference_model.eval()

# batch = next(iter(train_loader))
# for k,v in batch['chosen'].items():
#     print(k,v.shape)
# for k,v in batch['rejected'].items():
#     print(k,v.shape)

# print(policy_model(**batch['chosen']))
# print(policy_model(**batch['rejected']))

#policy_model.to(device)
#reference_model.to(device)

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
        logits=policy_model(**batch["chosen"]).logits,
        labels=batch["chosen"]['input_ids'],
        selection_mask=batch["chosen_mask"]
    )
    # log-probabilities of y_l given x from the language model
    policy_rejected_log_probas = compute_logprobs(
        logits=policy_model(**batch["rejected"]).logits,
        labels=batch["rejected"]['input_ids'],
        selection_mask=batch["rejected_mask"]
    )
    
    with torch.no_grad():
        # log-probabilities of y_w given x from the reference model
        ref_chosen_log_probas = compute_logprobs(
            logits=reference_model(**batch["chosen"]).logits,
            labels=batch["chosen"]['input_ids'],
            selection_mask=batch["chosen_mask"]
        )
        # log-probabilities of y_l given x from the reference model
        ref_rejected_log_probas = compute_logprobs(
            logits=reference_model(**batch["rejected"]).logits,
            labels=batch["rejected"]['input_ids'],
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
    for i, batch in enumerate(data_loader):
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

        else:
            break

    # calculate average
    total_loss /= num_batches
    total_chosen_rewards /= num_batches
    total_rejected_rewards /= num_batches
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

EPOCHS = 4
BETA = 0.1
steps = -1
# frequency at which the model will be evaluated
eval_freq = 1
# when the model is evaluated this variable
# decides the number of batches that will be used to calculate the metrics
eval_iter = 5

for epoch in range(EPOCHS):
    # set language model to training mode
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