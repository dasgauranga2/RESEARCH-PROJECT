from typing import Dict, List, Union, Tuple, Literal
import torch.distributed
from trl.trainer import DPOTrainer
from trl.trainer.utils import pad_to_length
import difflib

# custom class for training using mDPO
class mDPOTrainer(DPOTrainer):
    # method that concatenates both the chosen and rejected sequences along the batch dimension
    def concatenated_inputs(self, batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
        concatenated_batch = {}
        # finds the maximum length among chosen and rejected sequences
        if self.is_encoder_decoder:
            max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
        else:
            max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])
        
        # pad the chosen and rejected sequences (input_ids, attention mask, labels)
        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k or self.is_encoder_decoder else self.padding_value
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k or self.is_encoder_decoder else self.padding_value
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(self.accelerator.device)

        # create two copies of image for each chosen and rejected sequence
        concatenated_batch["concatenated_image"] = batch["image"] + batch["image"]

        if self.is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1)
            concatenated_batch["concatenated_attention_mask"] = batch["prompt_attention_mask"].repeat(2, 1)

        return concatenated_batch
    
    # performs forward pass on the model using a batch of data
    def concatenated_forward(
        self, model: torch.nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        # concatenate the chosen and rejected sequences (input_ids, attention mask, labels) along the batch dimension
        # the concatenated inputs will be given to the model together
        concatenated_batch = self.concatenated_inputs(batch)
        # length of chosen sequences which will be used 
        # later to separate the chosen and rejected sequences
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = {
            "images": concatenated_batch["concatenated_image"],
            "labels": concatenated_batch["concatenated_labels"],
        }
        # get the model outputs
        outputs, refined_labels = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            **model_kwargs,
        )
        all_logits = outputs.logits.to(torch.float32)

        # compute log-probabilities of the logits
        all_logps = self._get_batch_logps(
            all_logits,
            refined_labels,
            average_log_prob=False,
        )

        # splits the log-probabilities into chosen and rejected sequences
        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        # splits the logits into chosen and rejected sequences
        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        # repeat the above this time for the imageless part
        # this is done by corrupting the image
        imageless_model_kwargs = {
                "labels": batch["chosen_labels"],
                "images": batch["image"],
                "mask_visual_tokens": True,
            }
            
        imageless_chosen_outputs, imageless_chosen_label = model(
            batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
            **imageless_model_kwargs,
        )
        imageless_chosen_logits = imageless_chosen_outputs.logits.to(torch.float32)

        imageless_chosen_logps = self._get_batch_logps(
            imageless_chosen_logits,
            imageless_chosen_label,
            average_log_prob=False,
        )

        return (chosen_logps, rejected_logps, imageless_chosen_logps, chosen_logits, rejected_logits, imageless_chosen_logits)

    # method to calculate the mDPO loss
    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_imageless_chosen_logps: torch.FloatTensor, 
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_imageless_chosen_logps: torch.FloatTensor, 
        reference_free: bool = False,
    ):
        # log-probability ratio of chosen and rejected from the policy
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        # log-probability ratio of chosen and rejected from the reference
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        if reference_free:
            ref_logratios = 0

        # difference between log-probabilities of policy and reference
        logits = pi_logratios - ref_logratios  # response preference

        # repeat the same for corrupted image
        image_conditional_pi_logratios = policy_chosen_logps - policy_imageless_chosen_logps
        image_conditional_ref_logratios = reference_chosen_logps - reference_imageless_chosen_logps

        if reference_free:
            image_conditional_ref_logratios = 0

        image_conditional_logits = image_conditional_pi_logratios - image_conditional_ref_logratios  # image-conditional preference

        anchor_logits = policy_chosen_logps - reference_chosen_logps  # anchored preference

        # calculate the final mDPO loss
        losses = -torch.nn.functional.logsigmoid(self.beta * logits) \
            -torch.nn.functional.logsigmoid(self.beta * image_conditional_logits) \
            -torch.nn.functional.logsigmoid(self.beta * anchor_logits) 

        # calculate the rewards
        chosen_rewards = (
            self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        )
        rejected_rewards = (
            self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        )
        imageless_rewards = (
            self.beta * (policy_imageless_chosen_logps - reference_imageless_chosen_logps).detach()
        )

        return losses, chosen_rewards, rejected_rewards, imageless_rewards

    # return the final loss for gradient calculation and metrics
    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        # dictionary to store the metrics to be displayed
        metrics = {}
        # get the log-probabilities and logits of chosen and rejected sequences from the batch of data
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_imageless_chosen_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_imageless_chosen_logits,
        ) = self.concatenated_forward(model, batch)

        # repeat the same for reference model
        with torch.no_grad():
            if self.ref_model is None:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        reference_imageless_chosen_logps,
                        _,
                        _,
                        _,
                    ) = self.concatenated_forward(self.model, batch)
            else:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    reference_imageless_chosen_logps,
                    _,
                    _,
                    _,
                ) = self.concatenated_forward(self.ref_model, batch)

        # calculate the mDPO loss using the log-probabilities
        losses, chosen_rewards, rejected_rewards, imageless_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            policy_imageless_chosen_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            reference_imageless_chosen_logps,
        )

        # calculate the reward accuracies and margins
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        imageless_reward_accuracies = (chosen_rewards > imageless_rewards).float()

        # log the metrics
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics[f"{prefix}rewards/imageless_chosen"] = imageless_rewards.cpu().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/imageless_accuracies"] = imageless_reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics[f"{prefix}rewards/imageless_margins"] = (chosen_rewards - imageless_rewards).cpu().mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/imageless_chosen"] = policy_imageless_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().cpu().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().mean()
        metrics[f"{prefix}logits/imageless_chosen"] = policy_imageless_chosen_logits.detach().cpu().mean()

        # return the final scalar loss on which gradients will be calculated
        # and dictionary used for displaying metrics
        return losses.mean(), metrics
    
# custom class for training using mDPO with stable diffusion generated images
class mDPOSDTrainer(DPOTrainer):
    # method that concatenates both the chosen and rejected sequences along the batch dimension
    def concatenated_inputs(self, batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
        concatenated_batch = {}
        # finds the maximum length among chosen and rejected sequences
        if self.is_encoder_decoder:
            max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
        else:
            max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])
        
        # pad the chosen and rejected sequences (input_ids, attention mask, labels)
        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k or self.is_encoder_decoder else self.padding_value
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k or self.is_encoder_decoder else self.padding_value
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(self.accelerator.device)

        # create two copies of image for each chosen and rejected sequence
        concatenated_batch["concatenated_image"] = batch["original_image"] + batch["original_image"]

        if self.is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1)
            concatenated_batch["concatenated_attention_mask"] = batch["prompt_attention_mask"].repeat(2, 1)

        return concatenated_batch
    
    # performs forward pass on the model using a batch of data
    def concatenated_forward(
        self, model: torch.nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        # concatenate the chosen and rejected sequences (input_ids, attention mask, labels) along the batch dimension
        # the concatenated inputs will be given to the model together
        concatenated_batch = self.concatenated_inputs(batch)
        # length of chosen sequences which will be used 
        # later to separate the chosen and rejected sequences
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = {
            "images": concatenated_batch["concatenated_image"],
            "labels": concatenated_batch["concatenated_labels"],
        }
        # get the model outputs
        outputs, refined_labels = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            **model_kwargs,
        )
        all_logits = outputs.logits.to(torch.float32)

        # compute log-probabilities of the logits
        all_logps = self._get_batch_logps(
            all_logits,
            refined_labels,
            average_log_prob=False,
        )

        # splits the log-probabilities into chosen and rejected sequences
        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        # splits the logits into chosen and rejected sequences
        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        # repeat the above this time for the contrastive part
        # this is done by using the stable diffusion generated images
        chosen_sd_model_kwargs = {
                "labels": batch["chosen_labels"],
                "images": batch["chosen_image"]
            }
        rejected_sd_model_kwargs = {
                "labels": batch["chosen_labels"],
                "images": batch["rejected_image"]
            }
            
        chosen_sd_outputs, chosen_sd_label = model(
            batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
            **chosen_sd_model_kwargs,
        )
        rejected_sd_outputs, rejected_sd_label = model(
            batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
            **rejected_sd_model_kwargs,
        )
        chosen_sd_logits = chosen_sd_outputs.logits.to(torch.float32)
        rejected_sd_logits = rejected_sd_outputs.logits.to(torch.float32)

        chosen_sd_logps = self._get_batch_logps(
            chosen_sd_logits,
            chosen_sd_label,
            average_log_prob=False,
        )
        rejected_sd_logps = self._get_batch_logps(
            rejected_sd_logits,
            rejected_sd_label,
            average_log_prob=False,
        )

        return (chosen_logps, rejected_logps, chosen_sd_logps, rejected_sd_logps, chosen_logits, rejected_logits, chosen_sd_logits, rejected_sd_logits)

    # method to calculate the mDPO loss
    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_chosen_sd_logps: torch.FloatTensor, 
        policy_rejected_sd_logps: torch.FloatTensor, 
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_chosen_sd_logps: torch.FloatTensor, 
        reference_rejected_sd_logps: torch.FloatTensor, 
        reference_free: bool = False,
    ):
        # log-probability ratio of chosen and rejected from the policy
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        # log-probability ratio of chosen and rejected from the reference
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        if reference_free:
            ref_logratios = 0

        # difference between log-probabilities of policy and reference
        logits = pi_logratios - ref_logratios  # response preference

        # repeat the same for stable diffusion generated images
        image_conditional_pi_logratios = policy_chosen_sd_logps - policy_rejected_sd_logps
        image_conditional_ref_logratios = reference_chosen_sd_logps - reference_rejected_sd_logps

        if reference_free:
            image_conditional_ref_logratios = 0

        image_conditional_logits = image_conditional_pi_logratios - image_conditional_ref_logratios  # image-conditional preference

        anchor_logits = policy_chosen_logps - reference_chosen_logps  # anchored preference

        # calculate the final mDPO loss
        losses = -torch.nn.functional.logsigmoid(self.beta * logits) \
            -torch.nn.functional.logsigmoid(self.beta * image_conditional_logits) \
            -torch.nn.functional.logsigmoid(self.beta * anchor_logits) 

        # calculate the rewards
        chosen_rewards = (
            self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        )
        rejected_rewards = (
            self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        )
        chosen_sd_rewards = (
            self.beta * (policy_chosen_sd_logps - reference_chosen_sd_logps).detach()
        )
        rejected_sd_rewards = (
            self.beta * (policy_rejected_sd_logps - reference_rejected_sd_logps).detach()
        )

        return losses, chosen_rewards, rejected_rewards, chosen_sd_rewards, rejected_sd_rewards

    # return the final loss for gradient calculation and metrics
    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        # dictionary to store the metrics to be displayed
        metrics = {}
        # get the log-probabilities and logits of chosen and rejected sequences from the batch of data
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_sd_logps,
            policy_rejected_sd_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_chosen_sd_logits,
            policy_rejected_sd_logits,
        ) = self.concatenated_forward(model, batch)

        # repeat the same for reference model
        with torch.no_grad():
            if self.ref_model is None:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        reference_chosen_sd_logps,
                        reference_rejected_sd_logps,
                        _,
                        _,
                        _,
                        _,
                    ) = self.concatenated_forward(self.model, batch)
            else:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    reference_chosen_sd_logps,
                    reference_rejected_sd_logps,
                    _,
                    _,
                    _,
                    _,
                ) = self.concatenated_forward(self.ref_model, batch)

        # calculate the mDPO loss using the log-probabilities
        losses, chosen_rewards, rejected_rewards, chosen_sd_rewards, rejected_sd_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_sd_logps,
            policy_rejected_sd_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            reference_chosen_sd_logps,
            reference_rejected_sd_logps,
        )

        # calculate the reward accuracies and margins
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        sd_reward_accuracies = (chosen_sd_rewards > rejected_sd_rewards).float()

        # log the metrics
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics[f"{prefix}rewards/imageless1_chosen"] = chosen_sd_rewards.cpu().mean()
        metrics[f"{prefix}rewards/imageless2_chosen"] = rejected_sd_rewards.cpu().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/imageless_accuracies"] = sd_reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics[f"{prefix}rewards/imageless_margins"] = (chosen_sd_rewards - rejected_sd_rewards).cpu().mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/imageless1_chosen"] = policy_chosen_sd_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/imageless2_chosen"] = policy_rejected_sd_logps.detach().cpu().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().cpu().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().mean()
        metrics[f"{prefix}logits/imageless1_chosen"] = policy_chosen_sd_logits.detach().cpu().mean()
        metrics[f"{prefix}logits/imageless2_chosen"] = policy_rejected_sd_logits.detach().cpu().mean()

        # return the final scalar loss on which gradients will be calculated
        # and dictionary used for displaying metrics
        return losses.mean(), metrics
    
# custom class for training using mDPO with custom negative images
class mDPOCNITrainer(DPOTrainer):
    # method that concatenates both the chosen and rejected sequences along the batch dimension
    def concatenated_inputs(self, batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
        concatenated_batch = {}
        # finds the maximum length among chosen and rejected sequences
        if self.is_encoder_decoder:
            max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
        else:
            max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])
        
        # pad the chosen and rejected sequences (input_ids, attention mask, labels)
        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k or self.is_encoder_decoder else self.padding_value
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k or self.is_encoder_decoder else self.padding_value
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(self.accelerator.device)

        # create two copies of image for each chosen and rejected sequence
        concatenated_batch["concatenated_image"] = batch["original_image"] + batch["original_image"]

        if self.is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1)
            concatenated_batch["concatenated_attention_mask"] = batch["prompt_attention_mask"].repeat(2, 1)

        return concatenated_batch
    
    # performs forward pass on the model using a batch of data
    def concatenated_forward(
        self, model: torch.nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        # concatenate the chosen and rejected sequences (input_ids, attention mask, labels) along the batch dimension
        # the concatenated inputs will be given to the model together
        concatenated_batch = self.concatenated_inputs(batch)
        # length of chosen sequences which will be used 
        # later to separate the chosen and rejected sequences
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = {
            "images": concatenated_batch["concatenated_image"],
            "labels": concatenated_batch["concatenated_labels"],
        }
        # get the model outputs
        outputs, refined_labels = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            **model_kwargs,
        )
        all_logits = outputs.logits.to(torch.float32)

        # compute log-probabilities of the logits
        all_logps = self._get_batch_logps(
            all_logits,
            refined_labels,
            average_log_prob=False,
        )

        # splits the log-probabilities into chosen and rejected sequences
        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        # splits the logits into chosen and rejected sequences
        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        # repeat the above this time for the imageless part
        # this is done by corrupting the image
        imageless_model_kwargs = {
                "labels": batch["chosen_labels"],
                "images": batch["neg_image"]
            }
            
        imageless_chosen_outputs, imageless_chosen_label = model(
            batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
            **imageless_model_kwargs,
        )
        imageless_chosen_logits = imageless_chosen_outputs.logits.to(torch.float32)

        imageless_chosen_logps = self._get_batch_logps(
            imageless_chosen_logits,
            imageless_chosen_label,
            average_log_prob=False,
        )

        return (chosen_logps, rejected_logps, imageless_chosen_logps, chosen_logits, rejected_logits, imageless_chosen_logits)

    # method to calculate the mDPO loss
    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_imageless_chosen_logps: torch.FloatTensor, 
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_imageless_chosen_logps: torch.FloatTensor, 
        reference_free: bool = False,
    ):
        # log-probability ratio of chosen and rejected from the policy
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        # log-probability ratio of chosen and rejected from the reference
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        if reference_free:
            ref_logratios = 0

        # difference between log-probabilities of policy and reference
        logits = pi_logratios - ref_logratios  # response preference

        # repeat the same for corrupted image
        image_conditional_pi_logratios = policy_chosen_logps - policy_imageless_chosen_logps
        image_conditional_ref_logratios = reference_chosen_logps - reference_imageless_chosen_logps

        if reference_free:
            image_conditional_ref_logratios = 0

        image_conditional_logits = image_conditional_pi_logratios - image_conditional_ref_logratios  # image-conditional preference

        anchor_logits = policy_chosen_logps - reference_chosen_logps  # anchored preference

        # calculate the final mDPO loss
        losses = -torch.nn.functional.logsigmoid(self.beta * logits) \
            -torch.nn.functional.logsigmoid(self.beta * image_conditional_logits) \
            -torch.nn.functional.logsigmoid(self.beta * anchor_logits) 

        # calculate the rewards
        chosen_rewards = (
            self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        )
        rejected_rewards = (
            self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        )
        imageless_rewards = (
            self.beta * (policy_imageless_chosen_logps - reference_imageless_chosen_logps).detach()
        )

        return losses, chosen_rewards, rejected_rewards, imageless_rewards

    # return the final loss for gradient calculation and metrics
    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        # dictionary to store the metrics to be displayed
        metrics = {}
        # get the log-probabilities and logits of chosen and rejected sequences from the batch of data
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_imageless_chosen_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_imageless_chosen_logits,
        ) = self.concatenated_forward(model, batch)

        # repeat the same for reference model
        with torch.no_grad():
            if self.ref_model is None:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        reference_imageless_chosen_logps,
                        _,
                        _,
                        _,
                    ) = self.concatenated_forward(self.model, batch)
            else:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    reference_imageless_chosen_logps,
                    _,
                    _,
                    _,
                ) = self.concatenated_forward(self.ref_model, batch)

        # calculate the mDPO loss using the log-probabilities
        losses, chosen_rewards, rejected_rewards, imageless_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            policy_imageless_chosen_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            reference_imageless_chosen_logps,
        )

        # calculate the reward accuracies and margins
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        imageless_reward_accuracies = (chosen_rewards > imageless_rewards).float()

        # log the metrics
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics[f"{prefix}rewards/imageless_chosen"] = imageless_rewards.cpu().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/imageless_accuracies"] = imageless_reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics[f"{prefix}rewards/imageless_margins"] = (chosen_rewards - imageless_rewards).cpu().mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/imageless_chosen"] = policy_imageless_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().cpu().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().mean()
        metrics[f"{prefix}logits/imageless_chosen"] = policy_imageless_chosen_logits.detach().cpu().mean()

        # return the final scalar loss on which gradients will be calculated
        # and dictionary used for displaying metrics
        return losses.mean(), metrics
    
# custom class for training using DPA
class DPATrainer(DPOTrainer):
    def cal_batch_logp(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
    ) -> torch.FloatTensor:

        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        # decoder only model
        if not self.is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]

        labels[labels == self.label_pad_token_id] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        return per_token_logps
    
    def accumulate_logps(self, logps, signs):
        unique_signs, indices = torch.unique(signs, sorted=True, return_inverse=True)
        accumulated_logps = torch.zeros(signs.size(0), len(unique_signs) - 1, dtype=logps.dtype, device=logps.device)
        
        for i, sign in enumerate(unique_signs[1:]):
            mask = (signs == sign).float()
            accumulated_logps[:, i] = (logps * mask).sum(dim=-1)
        
        return accumulated_logps
    
    # function to expand signs to match the length of expanded labels
    def expand_signs(self, original_input_ids, original_signs, original_labels, expanded_labels,
                                        image_token_id=-200, ignore_index=-100):
        """
        Expand `signs` to align with expanded_labels, using original_input_ids and labels.

        Assumes:
            - Each sequence contains exactly ONE <image> token.
            - All samples expand to the same number of vision tokens.

        Returns:
            expanded_signs: Tensor [B, T_exp], aligned with expanded_labels.
        """
        B, T_exp = expanded_labels.shape

        # num_image_slots = expanded - (original - 1), because one <image> is replaced
        num_image_slots = expanded_labels.shape[1] - (original_labels.shape[1] - 1)

        expanded_signs = torch.full((B, T_exp), ignore_index, dtype=torch.long, device=original_input_ids.device)

        for b in range(B):
            exp_pos = 0
            for t, tok in enumerate(original_input_ids[b]):
                if tok.item() == image_token_id:
                    exp_pos += num_image_slots
                else:
                    if exp_pos < T_exp:
                        if t < len(original_signs[b]):
                            expanded_signs[b, exp_pos] = original_signs[b][t]
                        else:
                            expanded_signs[b, exp_pos] = ignore_index
                        exp_pos += 1
                        
        return expanded_signs

    def forward_batch(self, model, batch):
        """
        Run forward passes for chosen and rejected sequences and return
        log-probs, logits, labels, and signs for each branch.
        """

        # ---------------- chosen part ----------------
        model_kwargs = {
            "labels": batch["chosen_labels"],
            "images": batch["image"],
        }
        chosen_outputs, chosen_labels = model(
            batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
            **model_kwargs,
        )
        chosen_logits = chosen_outputs.logits.to(torch.float32)

        chosen_logps = self.cal_batch_logp(
            chosen_logits,
            chosen_labels,
        )
        
        chosen_expanded_signs = self.expand_signs(batch["chosen_input_ids"],
                                                  batch["chosen_signs"],
                                                  batch["chosen_labels"],
                                                  chosen_labels,
                                                  image_token_id=-200,
                                                  ignore_index=-100)

        # ---------------- rejected part ----------------
        model_kwargs = {
            "labels": batch["rejected_labels"],
            "images": batch["image"],
        }
        rejected_outputs, rejected_labels = model(
            batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"],
            **model_kwargs,
        )
        rejected_logits = rejected_outputs.logits.to(torch.float32)

        rejected_logps = self.cal_batch_logp(
            rejected_logits,
            rejected_labels,
        )

        rejected_expanded_signs = self.expand_signs(batch["rejected_input_ids"],
                                                    batch["rejected_signs"],
                                                    batch["rejected_labels"],
                                                    rejected_labels,
                                                    image_token_id=-200,
                                                    ignore_index=-100)

        # ---------------- return everything ----------------
        return (
            chosen_logps,
            rejected_logps,
            chosen_labels,
            rejected_labels,
            chosen_logits,
            rejected_logits,
            chosen_expanded_signs,
            rejected_expanded_signs,
        )

    # return the final loss for gradient calculation and metrics
    def get_batch_metrics(
        self,
        model,
        batch,
        train_eval: Literal["train", "eval"] = "train",
    ):  
        # dictionary to store the metrics to be displayed
        metrics = {}

        # -------------------- alignment loss --------------------
        (
            chosen_logps,
            rejected_logps,
            chosen_labels,
            rejected_labels,
            chosen_logits,
            rejected_logits,
            chosen_signs,
            rejected_signs,
        ) = self.forward_batch(model, batch)

        # masks (True for real tokens, False for -100)
        chosen_loss_mask = (chosen_labels[:, 1:] != -100).float()
        rejected_loss_mask = (rejected_labels[:, 1:] != -100).float()

        # zero-out logps on ignored positions
        chosen_logps = chosen_logps * chosen_loss_mask
        rejected_logps = rejected_logps * rejected_loss_mask

        chosen_signs   = chosen_signs[:, 1:]   # drop the first token to match L-1
        rejected_signs = rejected_signs[:, 1:]

        # signs: replace -100 with 0 so they can be used for grouping
        chosen_signs = chosen_signs.masked_fill(chosen_signs == -100, 0)
        rejected_signs = rejected_signs.masked_fill(rejected_signs == -100, 0)

        # phrase-level accumulation
        pos_logps_acc = self.accumulate_logps(chosen_logps, chosen_signs)   # [batch, num_spans_i]
        neg_logps_acc = self.accumulate_logps(rejected_logps, rejected_signs)

        # softplus(neg - pos) over spans, averaged
        alignment_loss = torch.log(1 + torch.exp(neg_logps_acc - pos_logps_acc))
        alignment_loss = alignment_loss.mean()

        # -------------------- divergence loss --------------------

        # forward pass with current policy
        #policy_logps, policy_labels, policy_logits = self.reference_forward(self.model, batch)
        (
            policy_logps,
            _,
            policy_labels,
            _,
            policy_logits,
            _,
            _,
            _,
        ) = self.forward_batch(model, batch)

        # forward pass with frozen reference model
        with torch.no_grad():
            #ref_logps, ref_labels, ref_logits = self.reference_forward(self.ref_model, batch)
            if self.ref_model is None:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    (
                        ref_logps,
                        _,
                        ref_labels,
                        _,
                        ref_logits,
                        _,
                        _,
                        _,
                    ) = self.forward_batch(self.model, batch)
            else:
                (
                    ref_logps,
                    _,
                    ref_labels,
                    _,
                    ref_logits,
                    _,
                    _,
                    _,
                ) = self.forward_batch(self.ref_model, batch)

        ref_loss_mask = (ref_labels != -100)
        vocab_size = ref_logits.shape[-1]

        ref_probs = torch.nn.functional.softmax(ref_logits, dim=-1)
        policy_probs = torch.nn.functional.softmax(policy_logits, dim=-1)

        divergence_loss = ref_probs * (ref_probs.log() - policy_probs.log())
        divergence_loss = divergence_loss * ref_loss_mask.unsqueeze(-1)
        divergence_loss = divergence_loss.sum() / divergence_loss.shape[0]   # normalize by batch size

        # -------------------- final loss --------------------
        alpha = 0.4   # weight for divergence loss
        loss = alignment_loss + alpha*divergence_loss
        
        # log the metrics
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}alignment_loss"] = alignment_loss.detach().cpu().mean()
        metrics[f"{prefix}divergence_loss"] = divergence_loss.detach().cpu().mean()
        metrics[f"{prefix}logps/chosen"] = chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/rejected"] = rejected_logps.detach().cpu().mean()
        metrics[f"{prefix}logits/chosen"] = chosen_logits.detach().cpu().mean()
        metrics[f"{prefix}logits/rejected"] = rejected_logits.detach().cpu().mean()

        return loss, metrics
    
# custom class for training using CHIP
class CHIPTrainer(DPOTrainer):
    def segment_loss(
        self,
        per_token_logps_chosen,      # (B, L_logits_c)  may include image tokens
        per_token_logps_rejected,    # (B, L_logits_r)
        chosen_labels,               # (B, L_labels)
        rejected_labels,             # (B, L_labels)
        dpo_token_weight: float = 4.0,
    ):
        """
        Segment-level contrastive loss that (1) aligns logits to the *shifted* label
        length (labels[:,1:]) and (2) upweights positions where chosen/rejected labels differ.
        """

        # --- 1) use shifted labels (the same shift used to compute per-token logps) ---
        chosen_labels_s   = chosen_labels[:, 1:].clone()
        rejected_labels_s = rejected_labels[:, 1:].clone()

        # --- 2) align logits len to labels len (truncate from the right if longer) ---
        T_c = chosen_labels_s.shape[1]
        T_r = rejected_labels_s.shape[1]

        if per_token_logps_chosen.shape[1] != T_c:
            per_token_logps_chosen = per_token_logps_chosen[:, -T_c:]
        if per_token_logps_rejected.shape[1] != T_r:
            per_token_logps_rejected = per_token_logps_rejected[:, -T_r:]

        # --- 3) valid masks (ignore padding -100) ---
        mask_c = (chosen_labels_s != -100)
        mask_r = (rejected_labels_s != -100)

        # --- 4) differing-token mask (only on positions that exist in both) ---
        # lengths T_c and T_r can differ; compare on the overlap region
        T = min(T_c, T_r)
        diff_same_len = (chosen_labels_s[:, :T] != rejected_labels_s[:, :T]) & mask_c[:, :T] & mask_r[:, :T]

        # initialize weights as 1.0 on valid tokens
        weight_c = mask_c.float()
        weight_r = mask_r.float()

        # upweight only the overlapping differing region
        weight_c[:, :T][diff_same_len] *= dpo_token_weight
        weight_r[:, :T][diff_same_len] *= dpo_token_weight

        # --- 5) weighted scores (normalize by effective weights) ---
        # (Make sure everything is on the same device / dtype)
        weight_c = weight_c.to(per_token_logps_chosen.dtype).to(per_token_logps_chosen.device)
        weight_r = weight_r.to(per_token_logps_rejected.dtype).to(per_token_logps_rejected.device)

        chosen_score   = (per_token_logps_chosen * weight_c).sum(-1) / weight_c.sum(-1).clamp(min=1)
        rejected_score = (per_token_logps_rejected * weight_r).sum(-1) / weight_r.sum(-1).clamp(min=1)

        # --- 6) logistic loss on the difference ---
        seg_logits = chosen_score - rejected_score
        seg_loss = -torch.nn.functional.logsigmoid(seg_logits)
        return seg_loss


    def compute_weighted_logp(self, per_token_logp, labels, token_weight, use_average=False):
        """
        Compute weighted log-probs for a sequence, aligning model outputs (which
        may include image tokens) with text labels/weights length.
        """
        # mask only text tokens (ignore padding -100)
        loss_mask = (labels[:, 1:].clone() != -100)

        # elementwise weighting
        weighted_mask = token_weight * loss_mask

        # 🔑 Align lengths: per_token_logp may include extra tokens (e.g., image tokens),
        # so truncate it from the right to match the weighted_mask length.
        if len(per_token_logp.shape) != 1:
            per_token_logp = per_token_logp[:, -weighted_mask.shape[1]:]

        logp = (per_token_logp * weighted_mask).sum(-1)

        average_logp = logp / weighted_mask.sum(-1)
        if use_average:
            return average_logp
        return logp

    
    def get_batch_logps(self, logits, labels, return_all=False):
        """
        Compute per-token and per-sequence log-probs.
        """
        assert logits.shape[:-1] == labels.shape

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = (labels != -100)

        # dummy token; we'll ignore losses here
        labels[labels == -100] = 0

        per_token_logps = torch.gather(
            logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
        ).squeeze(2)

        log_prob = (per_token_logps * loss_mask).sum(-1)
        avg_log_prob = log_prob / loss_mask.sum(-1)

        if return_all:
            return per_token_logps, log_prob, avg_log_prob

        return log_prob
    
    def chip_get_batch_logps(self,
                         logits: torch.FloatTensor,
                         reference_logits: torch.FloatTensor,
                         uncond_ref_logits: torch.FloatTensor,
                         labels: torch.LongTensor,
                         average_log_prob: bool = False):
        """
        Compute KL divergence and log probabilities like CHiP.
        """
        # Shift labels/logits to align
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        reference_logits = reference_logits[:, :-1, :]
        uncond_ref_logits = uncond_ref_logits[:, :-1, :]

        loss_mask = (labels != -100)
        labels[labels == -100] = 0

        vocab_logps = logits.log_softmax(-1)
        reference_vocab_ps = reference_logits.softmax(-1)
        reference_vocab_logps = reference_vocab_ps.log()
        uncond_ref_vocab_logps = uncond_ref_logits.log_softmax(-1)

        # Per-position KL
        per_position_kl = (reference_vocab_ps * (reference_vocab_logps - vocab_logps)).sum(-1)

        # Gather log probs for the target tokens
        per_policy_token_logps = torch.gather(vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
        per_reference_token_logps = torch.gather(reference_vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
        per_uncond_ref_token_logps = torch.gather(uncond_ref_vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_position_kl * loss_mask).sum(-1) / loss_mask.sum(-1), \
                (per_policy_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1), \
                (per_reference_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1), \
                (per_uncond_ref_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1), \
                per_policy_token_logps, per_reference_token_logps, per_uncond_ref_token_logps
        else:
            return (per_position_kl * loss_mask).sum(-1), \
                (per_policy_token_logps * loss_mask).sum(-1), \
                (per_reference_token_logps * loss_mask).sum(-1), \
                (per_uncond_ref_token_logps * loss_mask).sum(-1), \
                per_policy_token_logps, per_reference_token_logps, per_uncond_ref_token_logps

    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        metrics = {}

        # ==== 1. Forward pass for chosen/rejected (Response-Level) ====
        max_len = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        chosen_input_ids = pad_to_length(batch["chosen_input_ids"], max_len, pad_value=self.padding_value)
        rejected_input_ids = pad_to_length(batch["rejected_input_ids"], max_len, pad_value=self.padding_value)

        chosen_attention_mask = pad_to_length(batch["chosen_attention_mask"], max_len, pad_value=0)
        rejected_attention_mask = pad_to_length(batch["rejected_attention_mask"], max_len, pad_value=0)

        chosen_labels = pad_to_length(batch["chosen_labels"], max_len, pad_value=self.label_pad_token_id)
        rejected_labels = pad_to_length(batch["rejected_labels"], max_len, pad_value=self.label_pad_token_id)

        # Concatenate for a single forward pass
        batch["concatenated_input_ids"] = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)
        batch["concatenated_attention_mask"] = torch.cat([chosen_attention_mask, rejected_attention_mask], dim=0)
        batch["concatenated_labels"] = torch.cat([chosen_labels, rejected_labels], dim=0)
        batch["concatenated_images"] = batch["image"] + batch["image"]

        outputs, refined_labels = model(
            batch["concatenated_input_ids"],
            attention_mask=batch["concatenated_attention_mask"],
            images=batch["concatenated_images"],
            labels=batch["concatenated_labels"],
        )

        logits = outputs.logits.to(torch.float32)
        policy_logps = self._get_batch_logps(logits, refined_labels, average_log_prob=False)

        len_chosen = batch["chosen_labels"].shape[0]
        policy_chosen_logps = policy_logps[:len_chosen]
        policy_rejected_logps = policy_logps[len_chosen:]

        # ==== 2. Visual preference pass (chosen + corrupted image) ====
        visual_outputs, visual_labels = model(
            batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
            images=batch["corrupted_image"],
            labels=batch["chosen_labels"],
        )
        visual_logits = visual_outputs.logits.to(torch.float32)
        policy_visual_logps = self._get_batch_logps(visual_logits, visual_labels, average_log_prob=False)

        # ==== 3. Reference model forward ====
        with torch.no_grad():
            if self.ref_model is None:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_outputs, ref_labels = self.model(
                        batch["concatenated_input_ids"],
                        attention_mask=batch["concatenated_attention_mask"],
                        images=batch["image"] + batch["image"],
                        labels=batch["concatenated_labels"],
                    )
                    uncond_ref_outputs, uncond_ref_labels = self.model(
                        batch["concatenated_input_ids"],
                        attention_mask=batch["concatenated_attention_mask"],
                        images=batch["corrupted_image"] + batch["corrupted_image"],
                        labels=batch["concatenated_labels"],
                    )
            else:
                ref_outputs, ref_labels = self.ref_model(
                    batch["concatenated_input_ids"],
                    attention_mask=batch["concatenated_attention_mask"],
                    images=batch["image"] + batch["image"],
                    labels=batch["concatenated_labels"],
                )
                uncond_ref_outputs, uncond_ref_labels = self.ref_model(
                    batch["concatenated_input_ids"],
                    attention_mask=batch["concatenated_attention_mask"],
                    images=batch["corrupted_image"] + batch["corrupted_image"],
                    labels=batch["concatenated_labels"],
                )

        ref_logits = ref_outputs.logits.to(torch.float32)
        reference_logps = self._get_batch_logps(ref_logits, ref_labels, average_log_prob=False)
        reference_chosen_logps = reference_logps[:len_chosen]
        reference_rejected_logps = reference_logps[len_chosen:]

        uncond_ref_logits = uncond_ref_outputs.logits.to(torch.float32)
        uncond_ref_logps = self._get_batch_logps(uncond_ref_logits, uncond_ref_labels, average_log_prob=False)
        uncond_ref_win_logp, uncond_ref_rej_logp = uncond_ref_logps.split([len_chosen, len_chosen])

        # ==== 4. Token-level KL divergence ====
        all_position_kl, _, _, _, _, _, _ = self.chip_get_batch_logps(
            logits, ref_logits, uncond_ref_logits, refined_labels, average_log_prob=False
        )
        chosen_position_kl, rejected_position_kl = all_position_kl.split([len_chosen, len_chosen])

        # ==== 5. Segment-level contrastive loss ====
        per_token_logps, _, _ = self.get_batch_logps(logits, refined_labels, return_all=True)
        per_token_logps_chosen = per_token_logps[:len_chosen]
        per_token_logps_rejected = per_token_logps[len_chosen:]

        loss_segment = self.segment_loss(
            per_token_logps_chosen,
            per_token_logps_rejected,
            chosen_labels,
            rejected_labels,
            dpo_token_weight=4.0
        )

        # ==== 6. Aggregate CHiP loss ====
        logits_response = (policy_chosen_logps - policy_rejected_logps) - (
            reference_chosen_logps - reference_rejected_logps
        )
        logits_visual = (policy_chosen_logps - policy_visual_logps) - (
            reference_chosen_logps - uncond_ref_win_logp
        )
        logits_token = -(rejected_position_kl - chosen_position_kl.detach())

        beta, lambda_seg, gamma_tok = 0.5, 2.0, 0.1
        loss_response = -torch.nn.functional.logsigmoid(beta * logits_response)
        loss_visual = -torch.nn.functional.logsigmoid(beta * logits_visual)
        loss_token = -torch.nn.functional.logsigmoid(gamma_tok * logits_token)

        losses = loss_response + loss_visual + loss_segment + loss_token

        # ==== Rewards (logging only) ====
        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}loss"] = losses.mean().item()
        metrics[f"{prefix}response_loss"] = loss_response.mean().item()
        metrics[f"{prefix}visual_loss"] = loss_visual.mean().item()
        metrics[f"{prefix}segment_loss"] = loss_segment.mean().item()
        metrics[f"{prefix}token_loss"] = loss_token.mean().item()
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().item()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().item()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().item()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().item()

        return losses.mean(), metrics