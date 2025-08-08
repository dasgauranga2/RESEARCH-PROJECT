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
    
# custom class for training using mDPO with custom corrupted image
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
                "images": batch["corrupted_image"]
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
    
# custom class for training using mDPO with dual corrupted images
class mDPODCITrainer(DPOTrainer):
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
        imageless1_model_kwargs = {
                "labels": batch["chosen_labels"],
                "images": batch["corrupted_image1"]
            }
        imageless2_model_kwargs = {
                "labels": batch["chosen_labels"],
                "images": batch["corrupted_image2"]
            }
            
        imageless1_chosen_outputs, imageless1_chosen_label = model(
            batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
            **imageless1_model_kwargs,
        )
        imageless2_chosen_outputs, imageless2_chosen_label = model(
            batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
            **imageless2_model_kwargs,
        )
        imageless1_chosen_logits = imageless1_chosen_outputs.logits.to(torch.float32)
        imageless2_chosen_logits = imageless2_chosen_outputs.logits.to(torch.float32)

        imageless1_chosen_logps = self._get_batch_logps(
            imageless1_chosen_logits,
            imageless1_chosen_label,
            average_log_prob=False,
        )
        imageless2_chosen_logps = self._get_batch_logps(
            imageless2_chosen_logits,
            imageless2_chosen_label,
            average_log_prob=False,
        )

        return (chosen_logps, rejected_logps, imageless1_chosen_logps, imageless2_chosen_logps, chosen_logits, rejected_logits, imageless1_chosen_logits, imageless2_chosen_logits)

    # method to calculate the mDPO loss
    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_imageless1_chosen_logps: torch.FloatTensor, 
        policy_imageless2_chosen_logps: torch.FloatTensor, 
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_imageless1_chosen_logps: torch.FloatTensor, 
        reference_imageless2_chosen_logps: torch.FloatTensor, 
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
        image_conditional_pi_logratios = policy_imageless1_chosen_logps - policy_imageless2_chosen_logps
        image_conditional_ref_logratios = reference_imageless1_chosen_logps - reference_imageless2_chosen_logps

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
        imageless1_rewards = (
            self.beta * (policy_imageless1_chosen_logps - reference_imageless1_chosen_logps).detach()
        )
        imageless2_rewards = (
            self.beta * (policy_imageless2_chosen_logps - reference_imageless2_chosen_logps).detach()
        )

        return losses, chosen_rewards, rejected_rewards, imageless1_rewards, imageless2_rewards

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
            policy_imageless1_chosen_logps,
            policy_imageless2_chosen_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_imageless1_chosen_logits,
            policy_imageless2_chosen_logits,
        ) = self.concatenated_forward(model, batch)

        # repeat the same for reference model
        with torch.no_grad():
            if self.ref_model is None:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        reference_imageless1_chosen_logps,
                        reference_imageless2_chosen_logps,
                        _,
                        _,
                        _,
                        _,
                    ) = self.concatenated_forward(self.model, batch)
            else:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    reference_imageless1_chosen_logps,
                    reference_imageless2_chosen_logps,
                    _,
                    _,
                    _,
                    _,
                ) = self.concatenated_forward(self.ref_model, batch)

        # calculate the mDPO loss using the log-probabilities
        losses, chosen_rewards, rejected_rewards, imageless1_rewards, imageless2_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            policy_imageless1_chosen_logps,
            policy_imageless2_chosen_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            reference_imageless1_chosen_logps,
            reference_imageless2_chosen_logps,
        )

        # calculate the reward accuracies and margins
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        imageless_reward_accuracies = (imageless1_rewards > imageless2_rewards).float()

        # calculate other metrics
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics[f"{prefix}rewards/imageless1_chosen"] = imageless1_rewards.cpu().mean()
        metrics[f"{prefix}rewards/imageless2_chosen"] = imageless2_rewards.cpu().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/imageless_accuracies"] = imageless_reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics[f"{prefix}rewards/imageless_margins"] = (imageless1_rewards - imageless2_rewards).cpu().mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/imageless1_chosen"] = policy_imageless1_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/imageless2_chosen"] = policy_imageless2_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().cpu().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().mean()
        metrics[f"{prefix}logits/imageless1_chosen"] = policy_imageless1_chosen_logits.detach().cpu().mean()
        metrics[f"{prefix}logits/imageless2_chosen"] = policy_imageless2_chosen_logits.detach().cpu().mean()

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
                "labels": batch["chosen_summarized_labels"],
                "images": batch["chosen_image"]
            }
        rejected_sd_model_kwargs = {
                "labels": batch["hallucinated_labels"],
                "images": batch["rejected_image"]
            }
            
        chosen_sd_outputs, chosen_sd_label = model(
            batch["chosen_summarized_input_ids"],
            attention_mask=batch["chosen_summarized_attention_mask"],
            **chosen_sd_model_kwargs,
        )
        rejected_sd_outputs, rejected_sd_label = model(
            batch["hallucinated_input_ids"],
            attention_mask=batch["hallucinated_attention_mask"],
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
        image_conditional_pi_logratios = policy_chosen_sd_logps + policy_rejected_sd_logps
        image_conditional_ref_logratios = reference_chosen_sd_logps + reference_rejected_sd_logps

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
    
# custom class for training using Vanilla DPO
class VanillaDPOTrainer(DPOTrainer):
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

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    # method to calculate the Vanilla DPO loss
    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
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

        # calculate the final mDPO loss
        losses = -torch.nn.functional.logsigmoid(self.beta * logits)

        # calculate the rewards
        chosen_rewards = (
            self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        )
        rejected_rewards = (
            self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        )

        return losses, chosen_rewards, rejected_rewards

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
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)

        # repeat the same for reference model
        with torch.no_grad():
            if self.ref_model is None:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                    ) = self.concatenated_forward(self.model, batch)
            else:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                ) = self.concatenated_forward(self.ref_model, batch)

        # calculate the mDPO loss using the log-probabilities
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )

        # calculate the reward accuracies and margins
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        # calculate other metrics
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().cpu().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().mean()

        # return the final scalar loss on which gradients will be calculated
        # and dictionary used for displaying metrics
        return losses.mean(), metrics
    
# custom class for training using DPA
class DPATrainer(DPOTrainer):
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

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    # method to calculate the original DPO loss
    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
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

        # calculate the final DPO loss
        losses = -torch.nn.functional.logsigmoid(self.beta * logits) 

        # calculate the rewards
        chosen_rewards = (
            self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        )
        rejected_rewards = (
            self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        )

        return losses, chosen_rewards, rejected_rewards

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
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)

        # repeat the same for reference model
        with torch.no_grad():
            if self.ref_model is None:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                    ) = self.concatenated_forward(self.model, batch)
            else:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                ) = self.concatenated_forward(self.ref_model, batch)

        # calculate the DPO loss using the log-probabilities
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )

        # calculate the reward accuracies and margins
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        # calculate other metrics
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().cpu().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().mean()

        # return the final scalar loss on which gradients will be calculated
        # and dictionary used for displaying metrics
        return losses.mean(), metrics