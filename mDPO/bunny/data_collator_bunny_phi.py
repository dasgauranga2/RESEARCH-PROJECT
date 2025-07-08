from dataclasses import dataclass
from typing import Dict, List, Any
from trl.trainer.utils import DPODataCollatorWithPadding
from PIL import Image

from bunny_utils.util.mm_utils import tokenizer_image_token

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
    
# class used as a data collator to prepare and tokenize batches for mdpo with custom corrupted image
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

        for k, toks in {
            "chosen": chosen_sequence_tokens,
            "rejected": rejected_sequence_tokens,
            "prompt": prompt_tokens,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}_{type_key}"] = tokens

        # load the original image
        original_image = Image.open('./data/merged_images/' + img_path)
        # load the custom corrupted image
        corrupted_image = Image.open('./data/merged_images_corrupted/' + img_path)

        # process the image into a tensor
        original_image_tensor = self.model.process_images([original_image], self.model.config).to(dtype=self.model.dtype)
        corrupted_image_tensor = self.model.process_images([corrupted_image], self.model.config).to(dtype=self.model.dtype)
        batch["original_image"] = original_image_tensor
        batch["corrupted_image"] = corrupted_image_tensor

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
        #     "corrupted_image": <tensor representation of the image>  # image tensor of shape (1, channels, height, width)
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
    
# class used as a data collator to prepare and tokenize batches for mdpo with dual custom corrupted images
# it tokenizes text inputs, processes the images, creates the attention masks and labels
@dataclass
class mDPODCIDataCollatorBunny(DPODataCollatorWithPadding):
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

        for k, toks in {
            "chosen": chosen_sequence_tokens,
            "rejected": rejected_sequence_tokens,
            "prompt": prompt_tokens,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}_{type_key}"] = tokens

        # load the original image
        original_image = Image.open('./data/merged_images/' + img_path)
        # load the custom corrupted image
        corrupted_image1 = Image.open('./data/merged_images_corrupted1/' + img_path)
        corrupted_image2 = Image.open('./data/merged_images_corrupted2/' + img_path)

        # process the image into a tensor
        original_image_tensor = self.model.process_images([original_image], self.model.config).to(dtype=self.model.dtype)
        corrupted_image1_tensor = self.model.process_images([corrupted_image1], self.model.config).to(dtype=self.model.dtype)
        corrupted_image2_tensor = self.model.process_images([corrupted_image2], self.model.config).to(dtype=self.model.dtype)
        batch["original_image"] = original_image_tensor
        batch["corrupted_image1"] = corrupted_image1_tensor
        batch["corrupted_image2"] = corrupted_image2_tensor

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
        #     "corrupted_image1": <tensor representation of the image>  # image tensor of shape (1, channels, height, width)
        #     "corrupted_image2": <tensor representation of the image>  # image tensor of shape (1, channels, height, width)
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

# class used as a data collator to prepare and tokenize batches for dpa
# it tokenizes text inputs, processes the images, creates the attention masks and labels
@dataclass
class DPADataCollatorForDPO(DPODataCollatorWithPadding):
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
        image = Image.open('./data/' + img_path)
        # process the image into a tensor
        image_tensor = self.model.process_images([image], self.model.config).to(dtype=self.model.dtype)
        batch["image"] = image_tensor

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
            batch_element = self.tokenize_batch_element(feature["prompt"], feature["chosen"], feature["rejected"], feature["img_path"])
            tokenized_batch.append(batch_element)

        # collate the list of data points
        collated_batch = self.collate(tokenized_batch)
        return collated_batch
