from typing import Dict, List, Union, Tuple, Literal
import torch.distributed
from trl.trainer import DPOTrainer
from trl.trainer.utils import pad_to_length
from diff_lib import get_diff_ids

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
    # ---------------- utilities ----------------
    def cal_batch_logp(self, logits, labels, label_pad_token_id=-100, is_encoder_decoder=False):
        """Compute per-token log-probabilities."""
        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]

        labels = labels.clone()
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(
            logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
        ).squeeze(2)

        return per_token_logps

    def accumulate_logps(self, logps, signs):
        """
        Aggregate log-probs at phrase level using signs.
        - logps: [batch, seq_len]
        - signs: [batch, seq_len], integers (0 = non-span, >0 = span ID)
        """
        signs = signs[:, 1:].clone()  # ignore first token
        unique_signs = torch.unique(signs)
        phrase_ids = [s.item() for s in unique_signs if s.item() > 0]

        span_logps = []
        for span_id in phrase_ids:
            mask = (signs == span_id).float()
            span_logp = (logps * mask).sum(dim=-1)  # sum over tokens
            span_logps.append(span_logp.unsqueeze(1))

        if len(span_logps) == 0:
            return torch.zeros(logps.size(0), 0, device=logps.device)

        return torch.cat(span_logps, dim=1)

    def forward_batch(self, model, input_ids, labels, attention_mask, signs, images, label_pad_token_id=-100):
        """Run a forward pass and return per-token logps, logits, and refined labels."""
        outputs, refined_labels = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,          # raw labels, may be refined inside model
            images=images,
        )
        logits = outputs.logits.to(torch.float32)

        # use refined_labels for logps
        logps = self.cal_batch_logp(logits, refined_labels, label_pad_token_id=label_pad_token_id)
        return logps, logits, refined_labels

    def expand_signs(
        self,
        original_input_ids,
        original_signs,
        original_labels,
        expanded_labels,
        image_token_id=-200,
    ):
        """
        Expand `signs` to align with expanded_labels when image tokens are replaced
        by multiple visual embeddings (e.g., Bunny/SigLIP).

        Assumes:
            - Each sequence contains exactly ONE <image> token.
            - All samples expand to the same number of vision tokens.

        Returns:
            expanded_signs: Tensor [B, T_exp], aligned with expanded_labels.
        """
        B, T_exp = expanded_labels.shape
        T_orig = original_labels.shape[1]

        # number of new image tokens inserted
        num_image_slots = T_exp - (T_orig - 1)

        expanded_signs = torch.zeros(
            (B, T_exp), dtype=torch.long, device=original_input_ids.device
        )

        for b in range(B):
            exp_pos = 0
            for t, tok in enumerate(original_input_ids[b]):
                if tok.item() == image_token_id:
                    # fill visual token region with zeros (non-span)
                    fill_end = min(exp_pos + num_image_slots, T_exp)
                    expanded_signs[b, exp_pos:fill_end] = 0
                    exp_pos += num_image_slots
                else:
                    if exp_pos < T_exp:
                        val = (
                            original_signs[b][t]
                            if t < len(original_signs[b])
                            else 0
                        )
                        expanded_signs[b, exp_pos] = val
                        exp_pos += 1

        return expanded_signs


    # ---------------- metrics & loss ----------------
    def get_batch_metrics(
        self,
        model,
        batch,
        train_eval: Literal["train", "eval"] = "train",
    ):
        metrics = {}

        # -------- policy forward (chosen + rejected) --------
        chosen_logps, chosen_logits, chosen_labels = self.forward_batch(
            model,
            batch["chosen_input_ids"],
            batch["chosen_labels"],
            batch["chosen_attention_mask"],
            batch["chosen_signs"],
            batch["image"],
        )
        rejected_logps, rejected_logits, rejected_labels = self.forward_batch(
            model,
            batch["rejected_input_ids"],
            batch["rejected_labels"],
            batch["rejected_attention_mask"],
            batch["rejected_signs"],
            batch["image"],
        )

        # -------- expand signs --------
        batch["chosen_signs"] = self.expand_signs(
            batch["chosen_input_ids"],
            batch["chosen_signs"],
            batch["chosen_labels"],
            chosen_labels,
            image_token_id=-200
        )

        batch["rejected_signs"] = self.expand_signs(
            batch["rejected_input_ids"],
            batch["rejected_signs"],
            batch["rejected_labels"],
            rejected_labels,
            image_token_id=-200
        )

        # -------- accumulate logps --------
        pos_logps_acc = self.accumulate_logps(chosen_logps, batch["chosen_signs"])
        neg_logps_acc = self.accumulate_logps(rejected_logps, batch["rejected_signs"])
        alignment_loss = torch.log(1 + torch.exp(neg_logps_acc - pos_logps_acc)).mean()

        # -------- reference forward (chosen only) --------
        with torch.no_grad():
            if self.ref_model is None:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_chosen_logps, ref_chosen_logits, ref_chosen_labels = self.forward_batch(
                        self.model,
                        batch["chosen_input_ids"],
                        batch["chosen_labels"],
                        batch["chosen_attention_mask"],
                        batch["chosen_signs"],
                        batch["image"],
                    )
            else:
                ref_chosen_logps, ref_chosen_logits, ref_chosen_labels = self.forward_batch(
                    self.ref_model,
                    batch["chosen_input_ids"],
                    batch["chosen_labels"],
                    batch["chosen_attention_mask"],
                    batch["chosen_signs"],
                    batch["image"],
                )

        # -------- divergence loss (KL between ref and policy) --------
        ref_probs = ref_chosen_logits.softmax(dim=-1)
        policy_probs = chosen_logits.softmax(dim=-1)
        mask = ref_chosen_labels != self.label_pad_token_id

        divergence = (ref_probs * (ref_probs.log() - policy_probs.log()))
        divergence = divergence * mask.unsqueeze(-1)
        divergence = divergence.sum() / ref_probs.size(0)

        # -------- final loss --------
        loss = alignment_loss + 0.4*divergence

        # -------- metrics --------
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}alignment_loss"] = alignment_loss.detach().cpu().mean()
        metrics[f"{prefix}divergence_loss"] = divergence.detach().cpu().mean()
        metrics[f"{prefix}logps/chosen"] = chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/rejected"] = rejected_logps.detach().cpu().mean()
        metrics[f"{prefix}logits/chosen"] = chosen_logits.detach().cpu().mean()
        metrics[f"{prefix}logits/rejected"] = rejected_logits.detach().cpu().mean()

        return loss, metrics
    
# custom class for training using CHIP
class CHiPTrainer(DPOTrainer):
    # # function to calculate the segment-level loss
    # def segment_loss(
    #     self,
    #     per_token_logps_chosen,      # (B, L_logits_c)  may include image tokens
    #     per_token_logps_rejected,    # (B, L_logits_r)
    #     chosen_labels,               # (B, L_labels)
    #     rejected_labels,             # (B, L_labels)
    #     dpo_token_weight: float = 4.0,
    # ):
    #     """
    #     Segment-level contrastive loss that (1) aligns logits to the *shifted* label
    #     length (labels[:,1:]) and (2) upweights positions where chosen/rejected labels differ.
    #     """

    #     # --- 1) use shifted labels (the same shift used to compute per-token logps) ---
    #     chosen_labels_s   = chosen_labels[:, 1:].clone()
    #     rejected_labels_s = rejected_labels[:, 1:].clone()

    #     # --- 2) align logits len to labels len (truncate from the right if longer) ---
    #     T_c = chosen_labels_s.shape[1]
    #     T_r = rejected_labels_s.shape[1]

    #     if per_token_logps_chosen.shape[1] != T_c:
    #         per_token_logps_chosen = per_token_logps_chosen[:, -T_c:]
    #     if per_token_logps_rejected.shape[1] != T_r:
    #         per_token_logps_rejected = per_token_logps_rejected[:, -T_r:]

    #     # --- 3) valid masks (ignore padding -100) ---
    #     mask_c = (chosen_labels_s != -100)
    #     mask_r = (rejected_labels_s != -100)

    #     # --- 4) differing-token mask (only on positions that exist in both) ---
    #     # lengths T_c and T_r can differ; compare on the overlap region
    #     T = min(T_c, T_r)
    #     diff_same_len = (chosen_labels_s[:, :T] != rejected_labels_s[:, :T]) & mask_c[:, :T] & mask_r[:, :T]

    #     # initialize weights as 1.0 on valid tokens
    #     weight_c = mask_c.float()
    #     weight_r = mask_r.float()

    #     # upweight only the overlapping differing region
    #     weight_c[:, :T][diff_same_len] *= dpo_token_weight
    #     weight_r[:, :T][diff_same_len] *= dpo_token_weight

    #     # --- 5) weighted scores (normalize by effective weights) ---
    #     # (Make sure everything is on the same device / dtype)
    #     weight_c = weight_c.to(per_token_logps_chosen.dtype).to(per_token_logps_chosen.device)
    #     weight_r = weight_r.to(per_token_logps_rejected.dtype).to(per_token_logps_rejected.device)

    #     chosen_score   = (per_token_logps_chosen * weight_c).sum(-1) / weight_c.sum(-1).clamp(min=1)
    #     rejected_score = (per_token_logps_rejected * weight_r).sum(-1) / weight_r.sum(-1).clamp(min=1)

    #     # --- 6) logistic loss on the difference ---
    #     seg_logits = chosen_score - rejected_score
    #     seg_loss = -torch.nn.functional.logsigmoid(seg_logits)
    #     return seg_loss

    # function to calculate the segment-level action score
    def segment_action_score(
        self,
        per_token_logps,        # (B, L_logits)
        labels,                 # (B, L_labels)
        other_labels=None,      # (B, L_labels) from the opposite response, used to find differing tokens
        dpo_token_weight: float = 4.0,
    ):
        """
        Compute the segment-level action score A_seg for a *single* response under one model.

        Args:
            per_token_logps: per-token log probabilities from the model (B, L)
            labels: corresponding token labels (B, L)
            other_labels: optional second sequence (e.g., the opposite response)
                        used to identify differing tokens y_c between responses.
                        If None, weights = 1 everywhere (no upweighting).
            dpo_token_weight: gamma factor to upweight differing tokens.
        Returns:
            action_score: (B,) tensor of weighted average log-probabilities
                        i.e., A_seg(y|x,m)
        """

        # --- 1) Shift labels to align with next-token predictions ---
        labels_s = labels[:, 1:].clone()
        per_token_logps = per_token_logps[:, -labels_s.shape[1]:]

        # --- 2) Valid mask (ignore padding tokens) ---
        mask = (labels_s != -100)

        # --- 3) Compute differing-token mask if other_labels is provided ---
        if other_labels is not None:
            other_labels_s = other_labels[:, 1:].clone()
            T = min(labels_s.shape[1], other_labels_s.shape[1])
            diff_mask = (labels_s[:, :T] != other_labels_s[:, :T]) & mask[:, :T] & (other_labels_s[:, :T] != -100)
        else:
            # no comparison → no special weighting
            T = labels_s.shape[1]
            diff_mask = torch.zeros_like(mask[:, :T], dtype=torch.bool)

        # --- 4) Assign weights ---
        weight = mask.float()
        weight[:, :T][diff_mask] += dpo_token_weight

        # --- 5) Weighted average log-probability (segment-level action score) ---
        weight = weight.to(per_token_logps.dtype).to(per_token_logps.device)
        action_score = (per_token_logps * weight).sum(-1) / weight.sum(-1).clamp(min=1)

        return action_score
    
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

    # return the final loss for gradient calculation and metrics
    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        # dictionary to store the metrics to be displayed
        metrics = {}

        # find the maximum length among chosen and rejected sequences
        max_len = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        # pad the chosen and rejected sequences (input_ids, attention mask, labels)
        chosen_input_ids = pad_to_length(batch["chosen_input_ids"], max_len, pad_value=self.padding_value)
        rejected_input_ids = pad_to_length(batch["rejected_input_ids"], max_len, pad_value=self.padding_value)
        chosen_attention_mask = pad_to_length(batch["chosen_attention_mask"], max_len, pad_value=0)
        rejected_attention_mask = pad_to_length(batch["rejected_attention_mask"], max_len, pad_value=0)
        chosen_labels = pad_to_length(batch["chosen_labels"], max_len, pad_value=self.label_pad_token_id)
        rejected_labels = pad_to_length(batch["rejected_labels"], max_len, pad_value=self.label_pad_token_id)

        # concatenate the chosen and rejected sequences along the batch dimension
        batch["concatenated_input_ids"] = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)
        batch["concatenated_attention_mask"] = torch.cat([chosen_attention_mask, rejected_attention_mask], dim=0)
        batch["concatenated_labels"] = torch.cat([chosen_labels, rejected_labels], dim=0)
        batch["concatenated_images"] = batch["image"] + batch["image"]

        # pass the chosen and rejected responses with the original image to the model
        outputs, refined_labels = model(
            batch["concatenated_input_ids"],
            attention_mask=batch["concatenated_attention_mask"],
            images=batch["concatenated_images"],
            labels=batch["concatenated_labels"],
        )

        logits = outputs.logits.to(torch.float32)
        policy_logps = self._get_batch_logps(logits, refined_labels, average_log_prob=False)

        len_chosen = batch["chosen_labels"].shape[0]
        # log-probabilities of chosen response for the policy model
        policy_chosen_logps = policy_logps[:len_chosen]
        # log-probabilities of rejected response for the policy model
        policy_rejected_logps = policy_logps[len_chosen:]

        # pass the chosen response with the corrupted image to the model
        visual_outputs, visual_labels = model(
            batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
            images=batch["corrupted_image"],
            labels=batch["chosen_labels"],
        )
        visual_logits = visual_outputs.logits.to(torch.float32)
        # log-probabilities of chosen response with corrupted image for the policy model
        policy_visual_logps = self._get_batch_logps(visual_logits, visual_labels, average_log_prob=False)

        # reference model forward pass (frozen)
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
        # log-probabilities of chosen response for the reference model
        reference_chosen_logps = reference_logps[:len_chosen]
        # log-probabilities of rejected response for the reference model
        reference_rejected_logps = reference_logps[len_chosen:]

        uncond_ref_logits = uncond_ref_outputs.logits.to(torch.float32)
        uncond_ref_logps = self._get_batch_logps(uncond_ref_logits, uncond_ref_labels, average_log_prob=False)
        # log-probabilities of chosen response with corrupted image for the reference model
        uncond_ref_win_logp, _ = uncond_ref_logps.split([len_chosen, len_chosen])

        # calculates the per-position KL divergence 
        # between the policy and reference model
        all_position_kl, _, _, _, _, _, _ = self.chip_get_batch_logps(
            logits, ref_logits, uncond_ref_logits, refined_labels, average_log_prob=False
        )
        # split the KL divergence between policy and reference model
        # for chosen and rejected responses
        chosen_position_kl, rejected_position_kl = all_position_kl.split([len_chosen, len_chosen])

        # calculates per-token log-probabilities for 
        # chosen and rejected responses for policy
        per_token_logps, _, _ = self.get_batch_logps(logits, refined_labels, return_all=True)
        per_token_logps_chosen = per_token_logps[:len_chosen]
        per_token_logps_rejected = per_token_logps[len_chosen:]
        # calculates per-token log-probabilities for
        # chosen and rejected responses for reference
        ref_per_token_logps, _, _ = self.get_batch_logps(ref_logits, ref_labels, return_all=True)
        ref_per_token_logps_chosen = ref_per_token_logps[:len_chosen]
        ref_per_token_logps_rejected = ref_per_token_logps[len_chosen:]

        #beta, lambda_seg, gamma_tok = 0.5, 2.0, 0.1
        BETA = 0.5
        LAMBDA_SEG = 2.0
        GAMMA_TOK = 0.1
        
        # calculate segment-level action score for chosen response for policy
        sla_policy_chosen = self.segment_action_score(per_token_logps_chosen, chosen_labels, rejected_labels)
        # calculate segment-level action score for rejected response for policy
        sla_policy_rejected = self.segment_action_score(per_token_logps_rejected, rejected_labels, chosen_labels)
        # calculate segment-level action score for chosen response for reference
        sla_ref_chosen = self.segment_action_score(ref_per_token_logps_chosen, chosen_labels, rejected_labels)
        # calculate segment-level action score for rejected response for reference
        sla_ref_rejected = self.segment_action_score(ref_per_token_logps_rejected, rejected_labels, chosen_labels)

        # calculate the segment-level loss
        seg_logits = (sla_policy_chosen - sla_policy_rejected) - (sla_ref_chosen - sla_ref_rejected)
        loss_segment = -torch.nn.functional.logsigmoid(BETA * seg_logits)

        # calculate the response-level loss
        logits_response = (
            (policy_chosen_logps - policy_rejected_logps) 
            - (reference_chosen_logps - reference_rejected_logps)
            - GAMMA_TOK*(rejected_position_kl - chosen_position_kl.detach())
        )
        loss_response = -torch.nn.functional.logsigmoid(BETA * logits_response)

        # calculate the visual-preference loss
        logits_visual = (policy_chosen_logps - policy_visual_logps) - (
            reference_chosen_logps - uncond_ref_win_logp
        )
        loss_visual = -torch.nn.functional.logsigmoid(BETA * logits_visual)

        # aggregate all the losses
        losses = loss_response + loss_visual + LAMBDA_SEG*loss_segment

        # calculate the rewards
        chosen_rewards = BETA * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = BETA * (policy_rejected_logps - reference_rejected_logps).detach()
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        # log the metrics
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}loss"] = losses.mean().item()
        metrics[f"{prefix}response_loss"] = loss_response.mean().item()
        metrics[f"{prefix}visual_loss"] = loss_visual.mean().item()
        metrics[f"{prefix}segment_loss"] = loss_segment.mean().item()
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().item()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().item()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().item()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().item()

        return losses.mean(), metrics