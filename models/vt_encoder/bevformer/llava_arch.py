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


CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, image_features, image_sizes
    ):
        
        if  image_features is None or input_ids.shape[1] == 1:
            # if past_key_values is not None and image_features is not None and input_ids.shape[1] == 1:
            #     target_shape = past_key_values[-1][-1].shape[-2] + 1
            #     attention_mask = torch.cat((attention_mask, torch.ones(
            #         (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
            #         dtype=attention_mask.dtype,
            #         device=attention_mask.device
            #     )), dim=1)
            #     position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # Handle image_features reshaping more robustly
        # Check if image_features is already in the correct 3D shape with matching hidden_size
        if len(image_features.shape) == 3 and image_features.shape[2] == self.hidden_size:
            image_features = image_features.to(dtype=self.dtype)
        else:
            # Need to reshape: flatten and reshape to [batch_size, -1, hidden_size]
            batch_size = image_features.shape[0]
            # Flatten all dimensions except batch
            try:
                image_features_flat = image_features.view(batch_size, -1)
            except RuntimeError:
                # If view fails, use reshape which is more flexible
                image_features_flat = image_features.reshape(batch_size, -1)
            total_elements = image_features_flat.shape[1]
            
            # Check if total elements is divisible by hidden_size
            if total_elements % self.hidden_size != 0:
                # If not divisible, pad to make it divisible
                remainder = total_elements % self.hidden_size
                padding_size = self.hidden_size - remainder
                padding = torch.zeros(
                    batch_size, padding_size,
                    dtype=image_features_flat.dtype, 
                    device=image_features_flat.device
                )
                image_features_flat = torch.cat([image_features_flat, padding], dim=1)
            
            # Verify the total is now divisible
            total_elements_after_padding = image_features_flat.shape[1]
            assert total_elements_after_padding % self.hidden_size == 0, \
                f"Total elements ({total_elements_after_padding}) must be divisible by hidden_size ({self.hidden_size})"
            
            # Now reshape to [batch_size, num_tokens, hidden_size]
            num_tokens = image_features_flat.shape[1] // self.hidden_size
            image_features = image_features_flat.reshape(batch_size, num_tokens, self.hidden_size)
            image_features = image_features.to(dtype=self.dtype)

        # TODO: image start / end is not implemented here to support pretraining.
        # if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
        #     raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask.cpu()] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
            
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]

            #padding
            new_input_embeds_padded.append(torch.cat((
                cur_new_embed,
                torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
            ), dim=0))
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        # Handle position_ids: if originally None, set to None to let the model infer from inputs_embeds
        # This avoids shape mismatches during generation when using inputs_embeds
        if _position_ids is None:
            position_ids = None
        else:
            # If position_ids were provided, ensure they match the actual sequence length
            actual_seq_len = new_input_embeds.shape[1]
            if position_ids.shape[1] != actual_seq_len:
                # Recreate position_ids to match the actual sequence length
                position_ids_new = torch.zeros(
                    (batch_size, actual_seq_len), 
                    dtype=position_ids.dtype, 
                    device=position_ids.device
                )
                for i, (cur_new_embed, _) in enumerate(zip(new_input_embeds, new_labels)):
                    # Get the actual length from attention_mask or cur_new_embed
                    if attention_mask is not None:
                        cur_len = int(attention_mask[i].sum().item())
                    else:
                        # Fallback: use the full length
                        cur_len = actual_seq_len
                    if cur_len > 0:
                        position_ids_new[i, :cur_len] = torch.arange(
                            0, cur_len, 
                            dtype=position_ids.dtype, 
                            device=position_ids.device
                        )
                position_ids = position_ids_new

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
