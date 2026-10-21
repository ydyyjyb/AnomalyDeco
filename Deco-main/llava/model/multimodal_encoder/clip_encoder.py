import os
from pathlib import Path

import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


def _local_clip_candidates():
    here = Path(__file__).resolve()
    deco_main = here.parents[3]
    return [
        os.environ.get("VISION_TOWER_PATH", ""),
        str(deco_main / "clip-vit-large-patch14-336"),
        str(deco_main / "openai" / "clip-vit-large-patch14-336"),
        str(deco_main / "models" / "clip-vit-large-patch14-336"),
    ]


def _resolve_vision_tower_name(vision_tower):
    if vision_tower and os.path.exists(vision_tower):
        return vision_tower

    if vision_tower in ("openai/clip-vit-large-patch14-336", "clip-vit-large-patch14-336"):
        for cand in _local_clip_candidates():
            if cand and os.path.isfile(os.path.join(os.path.expanduser(cand), "config.json")):
                return os.path.expanduser(cand)

        raise FileNotFoundError(
            "LLaVA-v1.5-7b requires the CLIP vision tower openai/clip-vit-large-patch14-336, "
            "but no local copy was found and this server cannot reach HuggingFace.\n"
            "Please download/copy CLIP to one of these paths, or set VISION_TOWER_PATH:\n"
            "  Deco-main/clip-vit-large-patch14-336\n"
            "  Deco-main/openai/clip-vit-large-patch14-336\n"
            "  Deco-main/models/clip-vit-large-patch14-336\n"
            "Required files include config.json, preprocessor_config.json, and pytorch_model.bin/model.safetensors."
        )

    return vision_tower


# config "mm_vision_select_feature": "patch"
class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = _resolve_vision_tower_name(vision_tower)
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer] # 选择所取的特征层
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
        # print("image features shape")
        # print(image_features.shape)
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
