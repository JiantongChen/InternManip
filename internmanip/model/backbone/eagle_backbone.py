# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers.feature_extraction_utils import BatchFeature

import internmanip

from internmanip.model.backbone.eagle2_hg_model.inference_eagle_repo import reshape_model_embeddings

from .eagle2_hg_model.inference_eagle_repo import EagleProcessor, ModelSpecificValues

DEFAULT_EAGLE_MODEL_NAME = os.path.join(
    os.path.dirname(internmanip.__file__), "model", "backbone", "eagle2_hg_model"
)

DEFAULT_EAGLE_PATH = os.path.join(
    os.path.dirname(internmanip.__file__), "model", "backbone", "eagle2_hg_model_15"
)


def get_embeddings(
    self,
    reproject_vision: bool,
    pixel_values=None,
    input_ids=None,
    attention_mask=None,
    visual_features=None,
    output_hidden_states=None,
    skip_llm=False,
) -> torch.LongTensor:
    assert self.img_context_token_id is not None
    assert pixel_values is not None
    if visual_features is not None:
        vit_embeds = visual_features
    else:
        vit_embeds = self.extract_feature(pixel_values)
    input_embeds = self.language_model.get_input_embeddings()(input_ids)
    B, N, C = input_embeds.shape
    input_embeds = input_embeds.reshape(B * N, C)

    input_ids = input_ids.reshape(B * N)
    selected = input_ids == self.img_context_token_id
    assert selected.sum() != 0

    embeds_to_scatter = vit_embeds.reshape(-1, C).to(input_embeds.device, input_embeds.dtype)
    input_embeds[selected] = embeds_to_scatter
    input_embeds = input_embeds.reshape(B, N, C)

    # return hidden_states
    embeddings = self.language_model.forward(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    embeddings = embeddings.hidden_states[-1]

    if reproject_vision:
        embeddings = embeddings.reshape(B * N, C)
        embeddings[selected] = embeds_to_scatter
        embeddings = embeddings.reshape(B, N, C)
    return embeddings


class EagleBackbone(nn.Module):
    def __init__(
        self,
        select_layer: int = 12,
        model_name: str = DEFAULT_EAGLE_MODEL_NAME,
        tune_llm: bool = False,
        tune_visual: bool = False,
        reproject_vision: bool = False,
        scale_image_resolution: int = 1,
        processor_cfg: dict = None,
        projector_dim: int = -1,
        allow_reshape_visual: bool = True,
        remove_llm: bool = False,
        load_pretrained_det_eagle_path=None,
        eagle_path: str | None = None,
        use_flash_attention: bool = False,
        load_bf16: bool = False,
        project_to_dim: int = 1536,
        use_local_eagle_hg_model: bool = True,
    ):
        super().__init__()
        self.reproject_vision = reproject_vision


        # use local eagle model
        if use_local_eagle_hg_model:
            model_name = DEFAULT_EAGLE_MODEL_NAME

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_config(config, trust_remote_code=True)
        self.model.neftune_alpha = None

        if hasattr(self.model.vision_model, "vision_model") and hasattr(
            self.model.vision_model.vision_model, "head"
        ):
            self.model.vision_model.vision_model.head = torch.nn.Identity()

        # remove parts of the LLM
        self.model.language_model.lm_head = torch.nn.Identity()
        while len(self.model.language_model.model.layers) > select_layer:
            self.model.language_model.model.layers.pop(-1)

        # initialize processor
        processor = EagleProcessor(
            model_path=processor_cfg["model_path"],
            max_input_tiles=processor_cfg["max_input_tiles"],
            model_spec=ModelSpecificValues(**processor_cfg["model_spec"]),
        )
        self.model.img_context_token_id = processor.get_img_context_token()
        assert self.model.template == processor.model_spec.template
        assert self.model.num_image_token == processor.model_spec.num_image_token

        if projector_dim != -1:
            self.linear = torch.nn.Linear(projector_dim, 1536)
        else:
            self.linear = torch.nn.Identity()

        if allow_reshape_visual and scale_image_resolution != 1:
            reshape_model_embeddings(self.model, scale_image_resolution)

        if load_pretrained_det_eagle_path is not None:
            print("loading eagle model weight from: {}".format(load_pretrained_det_eagle_path))
            self.model.load_state_dict(torch.load(load_pretrained_det_eagle_path))

        self.set_trainable_parameters(tune_llm, tune_visual)

        if (
            hasattr(self.model, "vision_model")
            and hasattr(self.model.vision_model, "vision_model")
            and hasattr(self.model.vision_model.vision_model, "vision_towers")
            and len(self.model.vision_model.vision_model.vision_towers) > 1
        ):
            vision_towers = self.model.vision_model.vision_model.vision_towers

            if (
                hasattr(vision_towers[0], "vision_tower")
                and hasattr(vision_towers[0].vision_tower, "vision_model")
                and hasattr(vision_towers[0].vision_tower.vision_model, "encoder")
            ):
                vision_towers[0].vision_tower.vision_model.encoder.gradient_checkpointing = False
                vision_towers[0].vision_tower.vision_model.head = torch.nn.Identity()

            if hasattr(vision_towers[1], "vision_tower"):
                vision_towers[1].vision_tower.head = torch.nn.Identity()

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        if not tune_llm:
            self.model.language_model.requires_grad_(False)
        else:
            self.model.language_model.requires_grad_(True)

        if not tune_visual:
            self.model.vision_model.requires_grad_(False)
            self.model.mlp1.requires_grad_(False)
        else:
            self.model.vision_model.requires_grad_(True)
            self.model.mlp1.requires_grad_(True)

    def set_frozen_modules_to_eval_mode(self):
        if self.training:
            if self.model.language_model and not self.tune_llm:
                self.model.language_model.eval()
            if self.model.vision_model and not self.tune_visual:
                self.model.vision_model.eval()
                self.model.mlp1.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        # 0. Set frozen module to eval
        self.set_frozen_modules_to_eval_mode()

        embeddings = get_embeddings(
            self.model,
            self.reproject_vision,
            pixel_values=vl_input["pixel_values"],
            input_ids=vl_input["input_ids"],
            attention_mask=vl_input["attention_mask"],
        )

        embeddings = self.linear(embeddings)

        attention_mask = vl_input["attention_mask"]
        return BatchFeature(
            data={
                "backbone_features": embeddings,
                "backbone_attention_mask": attention_mask,
            }
        )  # [B, T2, hidden_size]



class EagleBackbone1_5(nn.Module):

    def __init__(
        self,
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = False,
        use_flash_attention: bool = False,
        load_bf16: bool = False,
        eagle_path: str | None = None,
        project_to_dim: int = 1536,
    ):
        """
        Args:
            tune_llm: whether to tune the LLM model (default: True)
            tune_visual: whether to tune the visual model (default: False)
        """
        super().__init__()
        assert not reproject_vision, "Reproject vision is not implemented here, set to False"

        config = AutoConfig.from_pretrained(DEFAULT_EAGLE_PATH, trust_remote_code=True)
        self.eagle_model = AutoModel.from_config(config, trust_remote_code=True)

        if project_to_dim is not None:
            self.eagle_linear = torch.nn.Linear(2048, project_to_dim)
        else:
            self.eagle_linear = torch.nn.Identity()

        # needed since we don't use these layers. Also saves compute
        while len(self.eagle_model.language_model.model.layers) > select_layer:
            self.eagle_model.language_model.model.layers.pop(-1)

        self.select_layer = select_layer
        self.set_trainable_parameters(tune_llm, tune_visual)

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            self.eagle_model.language_model.requires_grad_(False)
        if not tune_visual:
            self.eagle_model.vision_model.requires_grad_(False)
            self.eagle_model.mlp1.requires_grad_(False)
        print(f"Tune backbone llm: {self.tune_llm}")
        print(f"Tune backbone visual: {self.tune_visual}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_llm and not tune_visual:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Backbone trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No backbone trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if self.eagle_model.language_model and not self.tune_llm:
                self.eagle_model.language_model.eval()
            if self.eagle_model.vision_model and not self.tune_visual:
                self.eagle_model.vision_model.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward_eagle(self, vl_input: BatchFeature) -> BatchFeature:
        eagle_prefix = "eagle_"
        eagle_input = {
            k.removeprefix(eagle_prefix): v
            for k, v in vl_input.items()
            if k.startswith(eagle_prefix)
        }
        eagle_input.pop('image_sizes')

        eagle_output = self.eagle_model(**eagle_input, output_hidden_states=True, return_dict=True)
        eagle_features = eagle_output.hidden_states[self.select_layer]

        eagle_features = self.eagle_linear(eagle_features)
        return eagle_features, eagle_input["attention_mask"]

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()

        eagle_embeds, eagle_mask = self.forward_eagle(vl_input)
        return BatchFeature(
            data={"backbone_features": eagle_embeds, "backbone_attention_mask": eagle_mask}
        )  # [B, T2, hidden_size]
