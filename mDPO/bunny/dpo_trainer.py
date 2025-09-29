from typing import Dict, List, Union, Tuple, Literal
import torch.distributed
from trl.trainer import DPOTrainer
from trl.trainer.utils import pad_to_length

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

        # calculate other metrics
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

        # calculate other metrics
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

        # calculate other metrics
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

        # -------- pad signs to match logps length --------
        def align_signs(signs_list, target_len, device):
            signs_tensor = torch.zeros((len(signs_list), target_len), dtype=torch.long, device=device)
            for i, s in enumerate(signs_list):
                s = torch.tensor(s, device=device, dtype=torch.long)
                signs_tensor[i, :len(s)] = s
            return signs_tensor

        if not isinstance(batch["chosen_signs"], torch.Tensor):
            batch["chosen_signs"] = align_signs(batch["chosen_signs"], chosen_logps.size(1), chosen_logps.device)
        if not isinstance(batch["rejected_signs"], torch.Tensor):
            batch["rejected_signs"] = align_signs(batch["rejected_signs"], rejected_logps.size(1), rejected_logps.device)

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
        loss = alignment_loss + 0.1 * divergence

        # -------- metrics --------
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}alignment_loss"] = alignment_loss.detach().cpu().mean()
        metrics[f"{prefix}divergence_loss"] = divergence.detach().cpu().mean()
        metrics[f"{prefix}logps/chosen"] = chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/rejected"] = rejected_logps.detach().cpu().mean()
        metrics[f"{prefix}logits/chosen"] = chosen_logits.detach().cpu().mean()
        metrics[f"{prefix}logits/rejected"] = rejected_logits.detach().cpu().mean()

        return loss, metrics