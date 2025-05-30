import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional
import yaml
import datasets
import torch.distributed
import transformers
from accelerate.utils import DistributedType
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import GPTQConfig, deepspeed
from transformers.trainer_pt_utils import LabelSmoother

from modeling_bunny_phi import mDPOBunnyPhiForCausalLM
from data_collator_bunny_phi import mDPODataCollatorBunny, DPADataCollatorForDPO
from dpo_trainer import mDPOTrainer, DPATrainer, VanillaDPOTrainer

# run the script using the following command
# CUDA_VISIBLE_DEVICES=0 python bunny/run_dpo_bunny.py

# below classes with the '@dataclass' decorator is used
# to create class with special methods  
@dataclass
class ModelArguments:
    model_name_or_path: str = field(default=None)
    dataset_path: str = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False
    fix_vit: bool = True
    beta: float = field(default=0.1)
    generate_during_eval: bool = field(default=False)


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: str = ""
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False

# the three functions below are used when deepspeed is enabled
# they are used when saving the model using the 'safe_save_model_for_hf_trainer' function
# general utility function to gather parameters managed by DeepSpeed and ensure they are in a consistent state on the CPU
# used by the two functions below
def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.util.get_peft_model_state_dict
# function to collect PEFT-related parameters (with "lora_" in their names) utilizing maybe_zero_3 to process each parameter
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return

# function to collect non-PEFT parameters, optionally filtering for those that require gradients, and processes each parameter using maybe_zero_3
def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

# function to ensure that the model state is saved safely during training
def safe_save_model_for_hf_trainer(
    trainer: transformers.Trainer, output_dir: str, bias="none"
):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)


# def read_jsonl(file_path):
#     """Read a JSONL file and return a list of dictionaries."""
#     with open(file_path, "r", encoding="utf-8") as file:
#         return [json.loads(line) for line in file]

# function to find all the linear layers in a given model (excluding those associated with multimodal components)
# it returns a list of the names of these linear layers
def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            # print(_)
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

# function to train the model using mDPO
def train(config_dict):
    # variable used to indicate the rank of the current process in a distributed training setup
    global local_rank

    # use wandb for monitoring metrics with project name 'VLM_DPO'
    os.environ["WANDB_PROJECT"] = "VLM_DPO"
    # get arguments
    parser = transformers.HfArgumentParser(
        (ModelArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        training_args,
        lora_args,
    ) = parser.parse_dict(config_dict)

    # checks if both LoRA and deepspeed is enabled
    if getattr(training_args, "deepspeed", None) and getattr(
        lora_args, "q_lora", False
    ):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    # sets the local rank
    local_rank = training_args.local_rank

    device_map = None
    # determine if distributed data parallel is being used
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    # check if QLoRA is enabled
    # QLoRA extends LoRA by incorporating quantization techniques
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning("FSDP or ZeRO3 are not incompatible with QLoRA.")

    # Set RoPE scaling factor
    # get the configuration of the model
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        fp32=True,
    )
    # disable cache usage
    config.use_cache = False
    # set embedding dropout probability to zero
    config.embd_pdrop = 0

    # load the bunny model
    model = mDPOBunnyPhiForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map='auto',
        trust_remote_code=True,
        quantization_config=GPTQConfig(bits=4, disable_exllama=True)
        if training_args.use_lora and lora_args.q_lora
        else None,
    )
    # if LoRA is not enabled fix the vision transformer
    if not training_args.use_lora:
        if (
            training_args.fix_vit
            and hasattr(model, "transformer")
            and hasattr(model.transformer, "visual")
        ):
            model.transformer.visual.requires_grad_(False)
            if hasattr(model.transformer.visual, "attn_pool"):
                model.transformer.visual.attn_pool.requires_grad_(True)
    
    # load the tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    # set the padding token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # check if LoRA is enabled 
    if training_args.use_lora:
        # find the target modules to which LoRA will be applied
        if lora_args.lora_target_modules == "all-linear":
            lora_target_modules = find_all_linear_names(model)
        elif "," in lora_args.lora_target_modules:
            lora_target_modules = lora_args.lora_target_modules.split(",")
        else:
            lora_target_modules = lora_args.lora_target_modules
        
        # get the LoRA configuration
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
            # modules_to_save=None,  # This argument serves for adding new tokens.
        )
        # check if q-LoRA is enabled
        if lora_args.q_lora:
            # prepare the model for k-bit training
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
        # check if gradient checkpointing is enabled
        if training_args.gradient_checkpointing:
            # set the requires_grad attribute of input embeddings to True
            # this configuration is essential in scenarios where you need to compute gradients w.r.t. the input embeddings
            # such as when fine-tuning adapter layers like LoRA (Low-Rank Adaptation) while keeping the main model parameters frozen
            # by default, input embeddings in PyTorch models do not require gradients, meaning their requires_grad attribute is set to False
            # however, in certain training strategies, especially those involving parameter-efficient fine-tuning methods like LoRA
            # it's necessary to enable gradient computation for these embeddings to allow the gradients to propagate through the frozen layers to the trainable adapter layers
            model.enable_input_require_grads()

    # load the training dataset
    #print(model_args.dataset_path)
    # preference json data must be stored as mDPO/data/vlfeedback_llava_10k.json
    # image data must be stored as mDPO/data/merged_images/
    train_dataset = datasets.load_dataset('json', data_files=model_args.dataset_path, split="train")

    # print the number of trainable parameters of the model
    print_trainable_parameters(model)
    
    # custom trainer to train the model using mDPO
    trainer = VanillaDPOTrainer(
        model, # model to be trained
        args=training_args,
        beta=training_args.beta,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        data_collator=mDPODataCollatorBunny(
            tokenizer,
            model,
            max_length=training_args.model_max_length,
            max_prompt_length=training_args.model_max_length // 2,
            max_target_length=training_args.model_max_length // 2,
            label_pad_token_id=LabelSmoother.ignore_index,
            padding_value=tokenizer.pad_token_id,
            truncation_mode="keep_end",
        ), # custom data collator
        tokenizer=tokenizer, # model tokenizer
        max_length=training_args.model_max_length,
        peft_config=lora_config if training_args.use_lora else None,
        generate_during_eval=training_args.generate_during_eval,
    )

    # starts the training process without resuming from the last checkpoint
    trainer.train(resume_from_checkpoint=False)
    # saves the trainer state after training
    trainer.save_state()
    # saves the model configuration
    model.config.save_pretrained(training_args.output_dir)
    # saves the model for huggingface trainer
    safe_save_model_for_hf_trainer(trainer=trainer,
                                    output_dir=training_args.output_dir)


if __name__ == "__main__":
    with open("bunny/config.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    train(cfg)
