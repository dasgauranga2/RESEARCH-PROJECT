from dataclasses import dataclass
from typing import Dict, List, Any
from trl.trainer.utils import DPODataCollatorWithPadding
from PIL import Image
import random
from bunny_utils.util.mm_utils import tokenizer_image_token
import torch

# class used as a data collator to prepare and tokenize batches for mdpo
# it tokenizes text inputs, processes the images, creates the attention masks and labels
@dataclass
class mDPODataCollatorBunny(DPODataCollatorWithPadding):
    # processes a single data point
    def tokenize_batch_element(
        self,
        prompt: str, # prompt text
        chosen: str, # chosen response text
        rejected: str, # rejected response text
        img_path: str, # image path
    ) -> Dict:
        # dictionary to store the inputs
        batch = {}

        # tokenize the response texts
        chosen_tokens = self.tokenizer(chosen, add_special_tokens=False)
        rejected_tokens = self.tokenizer(rejected, add_special_tokens=False)

        prompt_tokens = {}
        # prompt token ids
        prompt_tokens["input_ids"] = tokenizer_image_token(prompt, self.tokenizer)
        # the attention mask helps the model differentiate between actual input tokens and padding tokens
        prompt_tokens["attention_mask"] = [1 for _ in range(len(prompt_tokens["input_ids"]))]

        # get the end-of-sequence token id
        eos_token_id = self.tokenizer.eos_token_id

        # the steps below adjust the attention mask by setting the position corresponding to exisiting EOS tokens to 0
        # this  ensures that the model does not attend to any tokens that come after the EOS token
        eos_indices_prompt = [i for i, x in enumerate(prompt_tokens["input_ids"]) if x == eos_token_id]
        # attention mask these indices to eos_token_id
        new_attention_mask = [
            0 if i in eos_indices_prompt else p for i, p in enumerate(prompt_tokens["attention_mask"])
        ]
        prompt_tokens["attention_mask"] = new_attention_mask

        # do the same for chosen and rejected responses
        eos_indices_chosen = [i for i, x in enumerate(chosen_tokens["input_ids"]) if x == eos_token_id]
        new_attention_mask_c = [
            0 if i in eos_indices_chosen else p for i, p in enumerate(chosen_tokens["attention_mask"])
        ]
        chosen_tokens["attention_mask"] = new_attention_mask_c

        eos_indices_rejected = [i for i, x in enumerate(rejected_tokens["input_ids"]) if x == eos_token_id]
        new_attention_mask_r = [
            0 if i in eos_indices_rejected else p for i, p in enumerate(rejected_tokens["attention_mask"])
        ]
        rejected_tokens["attention_mask"] = new_attention_mask_r

        # add EOS token to end of responses
        chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        chosen_tokens["attention_mask"].append(1)

        rejected_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        rejected_tokens["attention_mask"].append(1)

        # determine the longer of the chosen and rejected response
        longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

        # if combined sequence is too long, truncate the prompt
        if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
            if self.truncation_mode == "keep_start":
                prompt_tokens = {k: v[: self.max_prompt_length] for k, v in prompt_tokens.items()}
            elif self.truncation_mode == "keep_end":
                prompt_tokens = {k: v[-self.max_prompt_length :] for k, v in prompt_tokens.items()}
            else:
                raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

        # if that's still too long, truncate the response
        if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
            chosen_tokens = {k: v[: self.max_length - self.max_prompt_length] for k, v in chosen_tokens.items()}
            rejected_tokens = {
                k: v[: self.max_length - self.max_prompt_length] for k, v in rejected_tokens.items()
            }

        # concatenate the prompt and response tokens
        chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
        rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
        # labels are created from the above tokens such that
        # tokens corresponding to prompt tokens are masked 
        chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
        chosen_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
            prompt_tokens["input_ids"]
        )
        rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
        rejected_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
            prompt_tokens["input_ids"]
        )

        for k, toks in {
            "chosen": chosen_sequence_tokens,
            "rejected": rejected_sequence_tokens,
            "prompt": prompt_tokens,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}_{type_key}"] = tokens

        #print('./data/merged_images/' + img_path)
        image = Image.open('./data/merged_images/' + img_path)
        # process the image into a tensor
        image_tensor = self.model.process_images([image], self.model.config).to(dtype=self.model.dtype)
        batch["image"] = image_tensor

        # for k,v in batch.items():
        #     print(f"{k} DATA TYPE: {type(v)}\n")

        # the final result will be of this format
        #     batch = {
        #     "chosen_input_ids": [101, 2023, 2003, 1996, 2419, 3430, 1012, 102, 1],  # input token IDs for the prompt + chosen response
        #     "chosen_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1],  # attention mask
        #     "chosen_labels": [-100, -100, -100, -100, -100, -100, -100, 2023, 2003, 1996, 2419, 3430, 1012, 102, 1],  # labels for the prompt + chosen response with prompt part masked
        #     "rejected_input_ids": [101, 2023, 2003, 1996, 3446, 3430, 1012, 102, 1],  # input token IDs for the prompt + rejected response
        #     "rejected_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1],  # attention mask
        #     "rejected_labels": [-100, -100, -100, -100, -100, -100, -100, 2023, 2003, 1996, 3446, 3430, 1012, 102, 1],  # labels for the prompt + rejected response with prompt part masked
        #     "prompt_input_ids": [101, 2023, 2003, 2019, 2742, 3430, 2007, 2019, 999, 1],  # input token IDs for the prompt with image tokens
        #     "prompt_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # attention mask for the prompt
        #     "image": <tensor representation of the image>  # image tensor of shape (1, channels, height, width)
        # }

        return batch
    
    # processes a list of data points using the above method
    # and collates them into a single batch ready for model input
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            tokenized_batch = []

            for feature in features:
                prompt = feature["prompt"]
                chosen = feature["chosen"]
                rejected = feature["rejected"]
                img_path = feature["img_path"]

                batch_element = self.tokenize_batch_element(prompt, chosen, rejected, img_path)
                tokenized_batch.append(batch_element)

            # collate the list of data points
            collated_batch = self.collate(tokenized_batch)
            return collated_batch
    
# class used as a data collator to prepare and tokenize batches for mdpo with original image and stable diffusion generated images
# it tokenizes text inputs, processes the images, creates the attention masks and labels
@dataclass
class mDPOSDDataCollatorBunny(DPODataCollatorWithPadding):
    # processes a single data point
    def tokenize_batch_element(
        self,
        prompt: str, # prompt text
        chosen: str, # chosen response text
        rejected: str, # rejected response text
        chosen_summarized: str, # chosen summarized response text
        hallucinated: str, # hallucinated response text
        img_path: str, # image path
    ) -> Dict:
        # dictionary to store the inputs
        batch = {}

        # # random index to choose hallucinated answer/image
        # rhi = random.randint(0, 2)

        # tokenize the response texts
        chosen_tokens = self.tokenizer(chosen, add_special_tokens=False)
        rejected_tokens = self.tokenizer(rejected, add_special_tokens=False)
        chosen_summarized_tokens = self.tokenizer(chosen_summarized, add_special_tokens=False)
        hallucinated_tokens = self.tokenizer(hallucinated, add_special_tokens=False)

        prompt_tokens = {}
        # prompt token ids
        prompt_tokens["input_ids"] = tokenizer_image_token(prompt, self.tokenizer)
        # the attention mask helps the model differentiate between actual input tokens and padding tokens
        prompt_tokens["attention_mask"] = [1 for _ in range(len(prompt_tokens["input_ids"]))]

        # get the end-of-sequence token id
        eos_token_id = self.tokenizer.eos_token_id

        # the steps below adjust the attention mask by setting the position corresponding to exisiting EOS tokens to 0
        # this  ensures that the model does not attend to any tokens that come after the EOS token
        eos_indices_prompt = [i for i, x in enumerate(prompt_tokens["input_ids"]) if x == eos_token_id]
        # attention mask these indices to eos_token_id
        new_attention_mask = [
            0 if i in eos_indices_prompt else p for i, p in enumerate(prompt_tokens["attention_mask"])
        ]
        prompt_tokens["attention_mask"] = new_attention_mask

        # do the same for chosen and rejected responses
        eos_indices_chosen = [i for i, x in enumerate(chosen_tokens["input_ids"]) if x == eos_token_id]
        new_attention_mask_c = [
            0 if i in eos_indices_chosen else p for i, p in enumerate(chosen_tokens["attention_mask"])
        ]
        chosen_tokens["attention_mask"] = new_attention_mask_c

        eos_indices_rejected = [i for i, x in enumerate(rejected_tokens["input_ids"]) if x == eos_token_id]
        new_attention_mask_r = [
            0 if i in eos_indices_rejected else p for i, p in enumerate(rejected_tokens["attention_mask"])
        ]
        rejected_tokens["attention_mask"] = new_attention_mask_r

        eos_indices_chosen_summarized = [i for i, x in enumerate(chosen_summarized_tokens["input_ids"]) if x == eos_token_id]
        new_attention_mask_cs = [
            0 if i in eos_indices_chosen_summarized else p for i, p in enumerate(chosen_summarized_tokens["attention_mask"])
        ]
        chosen_summarized_tokens["attention_mask"] = new_attention_mask_cs

        eos_indices_hallucinated = [i for i, x in enumerate(hallucinated_tokens["input_ids"]) if x == eos_token_id]
        new_attention_mask_h = [
            0 if i in eos_indices_hallucinated else p for i, p in enumerate(hallucinated_tokens["attention_mask"])
        ]
        hallucinated_tokens["attention_mask"] = new_attention_mask_h

        # add EOS token to end of responses
        chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        chosen_tokens["attention_mask"].append(1)

        rejected_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        rejected_tokens["attention_mask"].append(1)

        chosen_summarized_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        chosen_summarized_tokens["attention_mask"].append(1)

        hallucinated_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        hallucinated_tokens["attention_mask"].append(1)

        # determine the longer of the chosen and rejected response
        longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

        # if combined sequence is too long, truncate the prompt
        if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
            if self.truncation_mode == "keep_start":
                prompt_tokens = {k: v[: self.max_prompt_length] for k, v in prompt_tokens.items()}
            elif self.truncation_mode == "keep_end":
                prompt_tokens = {k: v[-self.max_prompt_length :] for k, v in prompt_tokens.items()}
            else:
                raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

        # if that's still too long, truncate the response
        if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
            chosen_tokens = {k: v[: self.max_length - self.max_prompt_length] for k, v in chosen_tokens.items()}
            rejected_tokens = {
                k: v[: self.max_length - self.max_prompt_length] for k, v in rejected_tokens.items()
            }

        # concatenate the prompt and response tokens
        chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
        rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
        chosen_summarized_sequence_tokens = {k: prompt_tokens[k] + chosen_summarized_tokens[k] for k in chosen_summarized_tokens}
        hallucinated_sequence_tokens = {k: prompt_tokens[k] + hallucinated_tokens[k] for k in hallucinated_tokens}

        # lables are created from the above tokens such that
        # tokens corresponding to prompt tokens are masked 
        chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
        chosen_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
            prompt_tokens["input_ids"]
        )
        rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
        rejected_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
            prompt_tokens["input_ids"]
        )
        chosen_summarized_sequence_tokens["labels"] = chosen_summarized_sequence_tokens["input_ids"][:]
        chosen_summarized_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
            prompt_tokens["input_ids"]
        )
        hallucinated_sequence_tokens["labels"] = hallucinated_sequence_tokens["input_ids"][:]
        hallucinated_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
            prompt_tokens["input_ids"]
        )

        for k, toks in {
            "chosen": chosen_sequence_tokens,
            "rejected": rejected_sequence_tokens,
            "chosen_summarized": chosen_summarized_sequence_tokens,
            "hallucinated": hallucinated_sequence_tokens,
            "prompt": prompt_tokens,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}_{type_key}"] = tokens

        # load the original image
        original_image = Image.open('./data/merged_images/' + img_path)
        # load the stable diffusion generated images
        chosen_image = Image.open('./data/chosen/' + img_path)
        rejected_image = Image.open(f'./data/rejected/' + img_path)

        #print(f"IMAGE TYPE: {type(rejected_image)}\n")

        # process the image into a tensor
        original_image_tensor = self.model.process_images([original_image], self.model.config).to(dtype=self.model.dtype)
        chosen_image_tensor = self.model.process_images([chosen_image], self.model.config).to(dtype=self.model.dtype)
        rejected_image_tensor = self.model.process_images([rejected_image], self.model.config).to(dtype=self.model.dtype)
        batch["original_image"] = original_image_tensor
        batch["chosen_image"] = chosen_image_tensor
        batch["rejected_image"] = rejected_image_tensor

        # for k,v in batch.items():
        #     if isinstance(v, list):
        #         for ele in v:
        #             if not isinstance(ele, int):
        #                 print(f"ERROR WITH {k}:\n{v}")
        #                 raise

        # the final result will be of this format
        #     batch = {
        #     "chosen_input_ids": [101, 2023, 2003, 1996, 2419, 3430, 1012, 102, 1],  # input token IDs for the prompt + chosen response
        #     "chosen_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1],  # attention mask
        #     "chosen_labels": [-100, -100, -100, -100, -100, -100, -100, 2023, 2003, 1996, 2419, 3430, 1012, 102, 1],  # labels for the prompt + chosen response with prompt part masked
        #     "rejected_input_ids": [101, 2023, 2003, 1996, 3446, 3430, 1012, 102, 1],  # input token IDs for the prompt + rejected response
        #     "rejected_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1],  # attention mask
        #     "rejected_labels": [-100, -100, -100, -100, -100, -100, -100, 2023, 2003, 1996, 3446, 3430, 1012, 102, 1],  # labels for the prompt + rejected response with prompt part masked
        #     "prompt_input_ids": [101, 2023, 2003, 2019, 2742, 3430, 2007, 2019, 999, 1],  # input token IDs for the prompt with image tokens
        #     "prompt_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # attention mask for the prompt
        #     "original_image": <tensor representation of the image>  # image tensor of shape (1, channels, height, width)
        #     "chosen_image": <tensor representation of the image>  # image tensor of shape (1, channels, height, width)
        #     "rejected_image": <tensor representation of the image>  # image tensor of shape (1, channels, height, width)
        # }

        return batch
    
    # processes a list of data points using the above method
    # and collates them into a single batch ready for model input
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            tokenized_batch = []

            for feature in features:
                prompt = feature["prompt"]
                chosen = feature["chosen"]
                rejected = feature["rejected"]
                chosen_summarized = feature["chosen_summarized"]
                hallucinated = feature["hallucinated"]
                img_path = feature["img_path"]

                batch_element = self.tokenize_batch_element(prompt, 
                                                            chosen, rejected, 
                                                            chosen_summarized, hallucinated, 
                                                            img_path)
                tokenized_batch.append(batch_element)

            # collate the list of data points
            collated_batch = self.collate(tokenized_batch)
            return collated_batch

# class used as a data collator to prepare and tokenize batches for mdpo with original image and custom negative images
# it tokenizes text inputs, processes the images, creates the attention masks and labels
@dataclass
class mDPOCNIDataCollatorBunny(DPODataCollatorWithPadding):
    # processes a single data point
    def tokenize_batch_element(
        self,
        prompt: str, # prompt text
        chosen: str, # chosen response text
        rejected: str, # rejected response text
        img_path: str, # image path
    ) -> Dict:
        # dictionary to store the inputs
        batch = {}

        # tokenize the response texts
        chosen_tokens = self.tokenizer(chosen, add_special_tokens=False)
        rejected_tokens = self.tokenizer(rejected, add_special_tokens=False)

        prompt_tokens = {}
        # prompt token ids
        prompt_tokens["input_ids"] = tokenizer_image_token(prompt, self.tokenizer)
        # the attention mask helps the model differentiate between actual input tokens and padding tokens
        prompt_tokens["attention_mask"] = [1 for _ in range(len(prompt_tokens["input_ids"]))]

        # get the end-of-sequence token id
        eos_token_id = self.tokenizer.eos_token_id

        # the steps below adjust the attention mask by setting the position corresponding to exisiting EOS tokens to 0
        # this  ensures that the model does not attend to any tokens that come after the EOS token
        eos_indices_prompt = [i for i, x in enumerate(prompt_tokens["input_ids"]) if x == eos_token_id]
        # attention mask these indices to eos_token_id
        new_attention_mask = [
            0 if i in eos_indices_prompt else p for i, p in enumerate(prompt_tokens["attention_mask"])
        ]
        prompt_tokens["attention_mask"] = new_attention_mask

        # do the same for chosen and rejected responses
        eos_indices_chosen = [i for i, x in enumerate(chosen_tokens["input_ids"]) if x == eos_token_id]
        new_attention_mask_c = [
            0 if i in eos_indices_chosen else p for i, p in enumerate(chosen_tokens["attention_mask"])
        ]
        chosen_tokens["attention_mask"] = new_attention_mask_c

        eos_indices_rejected = [i for i, x in enumerate(rejected_tokens["input_ids"]) if x == eos_token_id]
        new_attention_mask_r = [
            0 if i in eos_indices_rejected else p for i, p in enumerate(rejected_tokens["attention_mask"])
        ]
        rejected_tokens["attention_mask"] = new_attention_mask_r

        # add EOS token to end of responses
        chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        chosen_tokens["attention_mask"].append(1)

        rejected_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        rejected_tokens["attention_mask"].append(1)

        # determine the longer of the chosen and rejected response
        longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

        # if combined sequence is too long, truncate the prompt
        if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
            if self.truncation_mode == "keep_start":
                prompt_tokens = {k: v[: self.max_prompt_length] for k, v in prompt_tokens.items()}
            elif self.truncation_mode == "keep_end":
                prompt_tokens = {k: v[-self.max_prompt_length :] for k, v in prompt_tokens.items()}
            else:
                raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

        # if that's still too long, truncate the response
        if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
            chosen_tokens = {k: v[: self.max_length - self.max_prompt_length] for k, v in chosen_tokens.items()}
            rejected_tokens = {
                k: v[: self.max_length - self.max_prompt_length] for k, v in rejected_tokens.items()
            }

        # concatenate the prompt and response tokens
        chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
        rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
        # labels are created from the above tokens such that
        # tokens corresponding to prompt tokens are masked 
        chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
        chosen_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
            prompt_tokens["input_ids"]
        )
        rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
        rejected_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
            prompt_tokens["input_ids"]
        )

        for k, toks in {
            "chosen": chosen_sequence_tokens,
            "rejected": rejected_sequence_tokens,
            "prompt": prompt_tokens,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}_{type_key}"] = tokens

        #print('./data/merged_images/' + img_path)
        original_image = Image.open('./data/merged_images/' + img_path)
        neg_image = Image.open('./data/rejected/' + img_path)
        # process the image into a tensor
        original_image_tensor = self.model.process_images([original_image], self.model.config).to(dtype=self.model.dtype)
        neg_image_tensor = self.model.process_images([neg_image], self.model.config).to(dtype=self.model.dtype)
        batch["original_image"] = original_image_tensor
        batch["neg_image"] = neg_image_tensor

        # for k,v in batch.items():
        #     print(f"{k} DATA TYPE: {type(v)}\n")

        # the final result will be of this format
        #     batch = {
        #     "chosen_input_ids": [101, 2023, 2003, 1996, 2419, 3430, 1012, 102, 1],  # input token IDs for the prompt + chosen response
        #     "chosen_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1],  # attention mask
        #     "chosen_labels": [-100, -100, -100, -100, -100, -100, -100, 2023, 2003, 1996, 2419, 3430, 1012, 102, 1],  # labels for the prompt + chosen response with prompt part masked
        #     "rejected_input_ids": [101, 2023, 2003, 1996, 3446, 3430, 1012, 102, 1],  # input token IDs for the prompt + rejected response
        #     "rejected_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1],  # attention mask
        #     "rejected_labels": [-100, -100, -100, -100, -100, -100, -100, 2023, 2003, 1996, 3446, 3430, 1012, 102, 1],  # labels for the prompt + rejected response with prompt part masked
        #     "prompt_input_ids": [101, 2023, 2003, 2019, 2742, 3430, 2007, 2019, 999, 1],  # input token IDs for the prompt with image tokens
        #     "prompt_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # attention mask for the prompt
        #     "image": <tensor representation of the image>  # image tensor of shape (1, channels, height, width)
        # }

        return batch
    
    # processes a list of data points using the above method
    # and collates them into a single batch ready for model input
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            tokenized_batch = []

            for feature in features:
                prompt = feature["prompt"]
                chosen = feature["chosen"]
                rejected = feature["rejected"]
                img_path = feature["img_path"]

                batch_element = self.tokenize_batch_element(prompt, chosen, rejected, img_path)
                tokenized_batch.append(batch_element)

            # collate the list of data points
            collated_batch = self.collate(tokenized_batch)
            return collated_batch
    
IGNORE_INDEX = -100
MASK_PLACEHOLDER_START = "<MASK>"
MASK_PLACEHOLDER_END = "</MASK>"

# class used as a data collator to prepare and tokenize batches for mdpo
# it tokenizes text inputs, processes the images, creates the attention masks and labels
@dataclass
class DPADataCollatorBunny(DPODataCollatorWithPadding):

    def tokenize_with_signs(self, prompt: str, answer_masked: str):
        """
        Tokenize a (prompt + masked answer) into input_ids, labels, and signs.

        Returns:
        - input_ids: token IDs for [prompt + answer + EOS]
        - attention_mask: 1 for real tokens, 0 for padding
        - labels: same as input_ids but with prompt positions set to IGNORE_INDEX
        - signs: span membership map aligned to input_ids
            0 = not part of any masked phrase (incl. prompt tokens)
            1, 2, ... = token belongs to masked phrase #1, #2, ...
        """
        # tokenize prompt (already contains <image>)
        prompt_tokens = tokenizer_image_token(prompt, self.tokenizer)
        prompt_attention = [1] * len(prompt_tokens)

        # tokenize answer with <MASK> tags
        tokens = []
        signs = []
        start_tag, end_tag = MASK_PLACEHOLDER_START, MASK_PLACEHOLDER_END
        span_id = 1
        start_index = 0

        while True:
            start_pos = answer_masked.find(start_tag, start_index)
            if start_pos == -1:
                # remaining unmasked piece
                piece = answer_masked[start_index:]
                if piece.strip():
                    piece_tokens = self.tokenizer(piece, add_special_tokens=False)["input_ids"]
                    tokens.extend(piece_tokens)
                    signs.extend([0] * len(piece_tokens))
                break

            # unmasked part before this mask
            piece = answer_masked[start_index:start_pos]
            if piece.strip():
                piece_tokens = self.tokenizer(piece, add_special_tokens=False)["input_ids"]
                tokens.extend(piece_tokens)
                signs.extend([0] * len(piece_tokens))

            # masked span itself
            end_pos = answer_masked.find(end_tag, start_pos)
            masked_piece = answer_masked[start_pos + len(start_tag): end_pos]
            if masked_piece.strip():
                piece_tokens = self.tokenizer(masked_piece, add_special_tokens=False)["input_ids"]
                tokens.extend(piece_tokens)
                signs.extend([span_id] * len(piece_tokens))
                span_id += 1

            start_index = end_pos + len(end_tag)

        # add EOS
        eos_id = self.tokenizer.eos_token_id
        tokens.append(eos_id)
        signs.append(0)

        # prepend zeros for prompt tokens so lengths match
        signs = [0] * len(prompt_tokens) + signs

        # combine
        input_ids = prompt_tokens + tokens
        attention_mask = prompt_attention + [1] * len(tokens)
        labels = [IGNORE_INDEX] * len(prompt_tokens) + tokens

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "signs": signs,
        }
    
    def convert_prompt(self, original_prompt):
        """
        Convert a DPA-style prompt into Bunny/mDPO-style conversation prompt.
        """
        bunny_prefix = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions. "
            "USER: "
        )
        bunny_suffix = " ASSISTANT:"
        return f"{bunny_prefix}{original_prompt}{bunny_suffix}"

    def tokenize_batch_element(self, prompt, chosen_masked, rejected_masked, img_path):
        batch = {}
        prompt = self.convert_prompt(prompt)

        # chosen (correct)
        chosen_dict = self.tokenize_with_signs(prompt, chosen_masked)
        for k, v in chosen_dict.items():
            batch[f"chosen_{k}"] = v

        # rejected (hallucinated)
        rejected_dict = self.tokenize_with_signs(prompt, rejected_masked)
        for k, v in rejected_dict.items():
            batch[f"rejected_{k}"] = v

        # image
        image = Image.open('./data/' + img_path).convert("RGB")
        image_tensor = self.model.process_images([image], self.model.config).to(dtype=self.model.dtype)
        batch["image"] = image_tensor

        # the final result will be of this format
        # batch = {
        #   # ----------------- Chosen (correct) sequence -----------------
        #   "chosen_input_ids": [101, 2003, 1037, 2158, 2009, 1012, 102],  
        #       # token IDs for the prompt + chosen (correct) response
        #
        #   "chosen_attention_mask": [1, 1, 1, 1, 1, 1, 1],  
        #       # 1 for real tokens, 0 for padding
        #
        #   "chosen_labels": [-100, -100, -100, 2158, 2009, 1012, 102],  
        #       # same as input_ids but with the prompt part masked (-100)
        #       # so the loss is only applied on the chosen answer
        #
        #   "chosen_signs": [0, 1, 1, 0, 0, 0, 0],  
        #       # span membership map aligned to tokens:
        #       #   0 = not part of any masked phrase
        #       #   1 = token belongs to masked phrase #1
        #       #   2 = token belongs to masked phrase #2, etc.
        #
        #   # ----------------- Rejected (hallucinated) sequence -----------------
        #   "rejected_input_ids": [101, 2003, 1037, 3899, 2009, 1012, 102],  
        #       # token IDs for the prompt + rejected (hallucinated) response
        #
        #   "rejected_attention_mask": [1, 1, 1, 1, 1, 1, 1],  
        #       # 1 for real tokens, 0 for padding
        #
        #   "rejected_labels": [-100, -100, -100, 3899, 2009, 1012, 102],  
        #       # prompt tokens masked out with -100
        #
        #   "rejected_signs": [0, 1, 1, 0, 0, 0, 0],  
        #       # span membership map for hallucinated spans
        #
        #   # ----------------- Image -----------------
        #   "image": <tensor representation of the image>
        #       # preprocessed image tensor of shape (1, channels, height, width)
        # }

        return batch

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        tokenized_batch = []
        for feature in features:
            prompt = feature["question"]
            chosen_masked = feature["correct_answer_masked"]
            rejected_masked = feature["hallucinated_answer_masked"]
            img_path = feature["image"]

            batch_element = self.tokenize_batch_element(prompt, chosen_masked, rejected_masked, img_path)
            tokenized_batch.append(batch_element)

        collated_batch = self.collate(tokenized_batch)
        return collated_batch
    
# class used as a data collator to prepare and tokenize batches for mdpo
# it tokenizes text inputs, processes the images, creates the attention masks and labels
@dataclass
class CHIPDataCollatorBunny(DPODataCollatorWithPadding):
    # add forward diffusion noise
    def add_diffusion_noise(self, image_tensor, noise_step):
        num_steps = 1000  # Number of diffusion steps

        # decide beta in each step
        betas = torch.linspace(-6, 6, num_steps)
        betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

        # decide alphas in each step
        alphas = 1 - betas
        alphas_prod = torch.cumprod(alphas, dim=0)
        alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)  # p for previous
        alphas_bar_sqrt = torch.sqrt(alphas_prod)
        one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
        one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

        def q_x(x_0, t):
            noise = torch.randn_like(x_0)
            alphas_t = alphas_bar_sqrt[t]
            alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
            return (alphas_t * x_0 + alphas_1_m_t * noise)

        noise_delta = int(noise_step)  # from 0-999
        noisy_image = image_tensor.clone()
        image_tensor_cd = q_x(noisy_image, noise_step)

        return image_tensor_cd

    # processes a single data point
    def tokenize_batch_element(
        self,
        prompt: str, # prompt text
        chosen: str, # chosen response text
        rejected: str, # rejected response text
        img_path: str, # image path
    ) -> Dict:
        # dictionary to store the inputs
        batch = {}

        # tokenize the response texts
        chosen_tokens = self.tokenizer(chosen, add_special_tokens=False)
        rejected_tokens = self.tokenizer(rejected, add_special_tokens=False)

        prompt_tokens = {}
        # prompt token ids
        prompt_tokens["input_ids"] = tokenizer_image_token(prompt, self.tokenizer)
        # the attention mask helps the model differentiate between actual input tokens and padding tokens
        prompt_tokens["attention_mask"] = [1 for _ in range(len(prompt_tokens["input_ids"]))]

        # get the end-of-sequence token id
        eos_token_id = self.tokenizer.eos_token_id

        # the steps below adjust the attention mask by setting the position corresponding to exisiting EOS tokens to 0
        # this  ensures that the model does not attend to any tokens that come after the EOS token
        eos_indices_prompt = [i for i, x in enumerate(prompt_tokens["input_ids"]) if x == eos_token_id]
        # attention mask these indices to eos_token_id
        new_attention_mask = [
            0 if i in eos_indices_prompt else p for i, p in enumerate(prompt_tokens["attention_mask"])
        ]
        prompt_tokens["attention_mask"] = new_attention_mask

        # do the same for chosen and rejected responses
        eos_indices_chosen = [i for i, x in enumerate(chosen_tokens["input_ids"]) if x == eos_token_id]
        new_attention_mask_c = [
            0 if i in eos_indices_chosen else p for i, p in enumerate(chosen_tokens["attention_mask"])
        ]
        chosen_tokens["attention_mask"] = new_attention_mask_c

        eos_indices_rejected = [i for i, x in enumerate(rejected_tokens["input_ids"]) if x == eos_token_id]
        new_attention_mask_r = [
            0 if i in eos_indices_rejected else p for i, p in enumerate(rejected_tokens["attention_mask"])
        ]
        rejected_tokens["attention_mask"] = new_attention_mask_r

        # add EOS token to end of responses
        chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        chosen_tokens["attention_mask"].append(1)

        rejected_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        rejected_tokens["attention_mask"].append(1)

        # determine the longer of the chosen and rejected response
        longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

        # if combined sequence is too long, truncate the prompt
        if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
            if self.truncation_mode == "keep_start":
                prompt_tokens = {k: v[: self.max_prompt_length] for k, v in prompt_tokens.items()}
            elif self.truncation_mode == "keep_end":
                prompt_tokens = {k: v[-self.max_prompt_length :] for k, v in prompt_tokens.items()}
            else:
                raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

        # if that's still too long, truncate the response
        if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
            chosen_tokens = {k: v[: self.max_length - self.max_prompt_length] for k, v in chosen_tokens.items()}
            rejected_tokens = {
                k: v[: self.max_length - self.max_prompt_length] for k, v in rejected_tokens.items()
            }

        # concatenate the prompt and response tokens
        chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
        rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
        # labels are created from the above tokens such that
        # tokens corresponding to prompt tokens are masked 
        chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
        chosen_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
            prompt_tokens["input_ids"]
        )
        rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
        rejected_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
            prompt_tokens["input_ids"]
        )

        for k, toks in {
            "chosen": chosen_sequence_tokens,
            "rejected": rejected_sequence_tokens,
            "prompt": prompt_tokens,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}_{type_key}"] = tokens

        #print('./data/merged_images/' + img_path)
        image = Image.open('./data/rlhf_images/' + img_path)
        # process the image into a tensor
        image_tensor = self.model.process_images([image], self.model.config).to(dtype=self.model.dtype)
        # create corrupted image
        corrupted_image_tensor = self.add_diffusion_noise(image_tensor, 500)
        batch["image"] = image_tensor
        batch["corrupted_image"] = corrupted_image_tensor

        # for k,v in batch.items():
        #     print(f"{k} DATA TYPE: {type(v)}\n")

        # the final result will be of this format
        #     batch = {
        #     "chosen_input_ids": [101, 2023, 2003, 1996, 2419, 3430, 1012, 102, 1],  # input token IDs for the prompt + chosen response
        #     "chosen_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1],  # attention mask
        #     "chosen_labels": [-100, -100, -100, -100, -100, -100, -100, 2023, 2003, 1996, 2419, 3430, 1012, 102, 1],  # labels for the prompt + chosen response with prompt part masked
        #     "rejected_input_ids": [101, 2023, 2003, 1996, 3446, 3430, 1012, 102, 1],  # input token IDs for the prompt + rejected response
        #     "rejected_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1],  # attention mask
        #     "rejected_labels": [-100, -100, -100, -100, -100, -100, -100, 2023, 2003, 1996, 3446, 3430, 1012, 102, 1],  # labels for the prompt + rejected response with prompt part masked
        #     "prompt_input_ids": [101, 2023, 2003, 2019, 2742, 3430, 2007, 2019, 999, 1],  # input token IDs for the prompt with image tokens
        #     "prompt_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # attention mask for the prompt
        #     "image": <tensor representation of the image>  # image tensor of shape (1, channels, height, width)
        #     "corrupted_image": <tensor representation of the corrupted image>  # corrupted image tensor of shape (1, channels, height, width)
        # }

        return batch
    
    # processes a list of data points using the above method
    # and collates them into a single batch ready for model input
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            tokenized_batch = []

            for feature in features:
                prompt = feature["prompt"]
                chosen = feature["chosen"]
                rejected = feature["rejected"]
                img_path = feature["img_path"]

                batch_element = self.tokenize_batch_element(prompt, chosen, rejected, img_path)
                tokenized_batch.append(batch_element)

            # collate the list of data points
            collated_batch = self.collate(tokenized_batch)
            return collated_batch