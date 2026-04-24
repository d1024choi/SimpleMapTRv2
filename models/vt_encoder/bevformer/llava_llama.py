# Copyright (c) 2024-2025, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia License.
# To view a copy of this license, visit
# https://github.com/NVlabs/OmniDrive/blob/main/LICENSE
#
# SPDX-License-Identifier: Apache-2.0
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from models.vt_encoder.bevformer.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.hidden_size = config.hidden_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.pretraining_tp = config.pretraining_tp

        number_tokens = [
                718,
                448,
                29900,
                29889,
                29896,
                29906,
                29941,
                29946,
                29945,
                29953,
                29955,
                29947,
                29929,
            ]  # +-0.123456789
        weighted_mask = torch.ones(self.config.vocab_size)
        weighted_mask[number_tokens] = 3.0
        self.register_buffer("weighted_mask", weighted_mask)
        
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass for LLaVA-Llama vision-language model.
        
        This method processes multimodal inputs (text tokens and images) and performs
        causal language modeling: predicting the next token in a sequence given previous tokens
        and optional visual context.
        
        Args:
            input_ids: Token IDs for text input (shape: [batch, seq_len])
            attention_mask: Mask indicating which tokens to attend to
            position_ids: Explicit position embeddings (optional, auto-generated if None)
            past_key_values: Cached key-value pairs from previous forward passes (for generation)
            inputs_embeds: Pre-computed token embeddings (if provided, skips embedding lookup)
            labels: Ground truth token IDs for computing loss (shape: [batch, seq_len])
            use_cache: Whether to cache key-value pairs for faster generation
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states from all layers
            images: Vision features from image encoder (shape: [batch, num_images, ...])
            image_sizes: Original image sizes for proper feature processing
            return_dict: Whether to return a dict or tuple
        
        Returns:
            CausalLMOutputWithPast: Contains logits, loss, and optional hidden states/attentions
        """
        # Prepare multimodal inputs: combine text tokens with vision embeddings
        # This method processes images through vision encoder and interleaves vision tokens
        # with text tokens, creating a unified sequence that the language model can process
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        # Use provided flags or fall back to config defaults
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward through the LLaMA transformer model
        # This processes the combined text+vision embeddings through multiple transformer layers
        # Outputs include hidden states and optional attention weights/past key-values
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Extract hidden states from the last layer (outputs[0] is the main hidden states)
        hidden_states = outputs[0]
        
        # Project hidden states to vocabulary logits for next-token prediction
        # Handle tensor parallelism: if model was trained with tensor parallelism, split the
        # language model head weights and compute logits in parallel, then concatenate
        if self.pretraining_tp > 1:
            # Split the vocabulary into chunks for parallel processing
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            # Standard case: single linear projection from hidden size to vocabulary size
            logits = self.lm_head(hidden_states)
        logits = logits.float()  # Convert to float32 for numerical stability in loss computation

        # Compute cross-entropy loss if labels are provided (training mode)
        loss = None
        if labels is not None:
            # Causal LM prediction: token at position i predicts token at position i+1
            # Shift logits and labels so that logits[i] corresponds to labels[i+1]
            # This aligns predictions with the next token in the sequence
            shift_logits = logits[..., :-1, :].contiguous()  # Remove last token (no prediction for it)
            shift_labels = labels[..., 1:].contiguous()  # Remove first token (no label for it)
            
            # Flatten sequences and vocabulary dimensions for loss computation
            # Shape: [batch * seq_len, vocab_size] and [batch * seq_len]
            loss_fct = CrossEntropyLoss(weight=self.weighted_mask.float())  # Weighted loss (higher weight for number tokens)
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            
            # Ensure labels are on the same device as logits (for multi-GPU setups)
            shift_labels = shift_labels.to(shift_logits.device)
            
            # Compute cross-entropy loss between predicted logits and ground truth labels
            loss = loss_fct(shift_logits, shift_labels)
            loss = torch.nan_to_num(loss)  # Replace NaN values with 0 (safety measure)

        # Return results in the requested format (dict or tuple)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        # Return structured output containing all relevant information
        return CausalLMOutputWithPast(
            loss=loss,  # Computed loss (None if labels not provided)
            logits=logits,  # Predicted logits for each token position
            past_key_values=outputs.past_key_values,  # Cached key-values for generation
            hidden_states=outputs.hidden_states,  # Hidden states from all layers (if requested)
            attentions=outputs.attentions,  # Attention weights (if requested)
        )
       

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
