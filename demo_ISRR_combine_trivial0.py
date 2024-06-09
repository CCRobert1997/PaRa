import os
try:
    os.chdir('sliders')
except:
    pass
import torch
from PIL import Image
import argparse
import os, json, random
import pandas as pd
import matplotlib.pyplot as plt
import glob, re

from safetensors.torch import load_file
import matplotlib.image as mpimg
import copy
import gc
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import DiffusionPipeline
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel, LMSDiscreteScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor, AttentionProcessor
from typing import Any, Dict, List, Optional, Tuple, Union
from trainscripts.textsliders.lora import LoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput


import math
from typing import Optional, List, Type, Set, Literal

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from safetensors.torch import save_file













import inspect
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from diffusers.pipelines import StableDiffusionXLPipeline
import random

import torch
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

def flush():
    torch.cuda.empty_cache()
    gc.collect()
    
@torch.no_grad()
def call(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
    
        network=None,
        start_noise=None,
        scale=None,
        unet=None,
    ):
        
        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
            )
        else:
            negative_add_time_ids = add_time_ids

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 7.1 Apply denoising_end
        if denoising_end is not None and isinstance(denoising_end, float) and denoising_end > 0 and denoising_end < 1:
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]
        latents = latents.to(unet.dtype)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if t>start_noise:
                    network.set_lora_slider(scale=0)
                else:
                    network.set_lora_slider(scale=scale)
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                with network:
                    noise_pred = unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
#         self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)










device = 'cuda'
StableDiffusionXLPipeline.__call__ = call



UNET_TARGET_REPLACE_MODULE_TRANSFORMER = [
#     "Transformer2DModel",  # どうやらこっちの方らしい？ # attn1, 2
    "Attention"
]
UNET_TARGET_REPLACE_MODULE_CONV = [
    "ResnetBlock2D",
    "Downsample2D",
    "Upsample2D",
    "DownBlock2D",
    "UpBlock2D",
    
]  # locon, 3clier

LORA_PREFIX_UNET = "lora_unet"

DEFAULT_TARGET_REPLACE = UNET_TARGET_REPLACE_MODULE_TRANSFORMER

TRAINING_METHODS = Literal[
    "noxattn",  # train all layers except x-attns and time_embed layers
    "innoxattn",  # train all layers except self attention layers
    "selfattn",  # ESD-u, train only self attention layers
    "xattn",  # ESD-x, train only x attention layers
    "full",  #  train all layers
    "xattn-strict", # q and k values
    "noxattn-hspace",
    "noxattn-hspace-last",
    # "xlayer",
    # "outxattn",
    # "outsattn",
    # "inxattn",
    # "inmidsattn",
    # "selflayer",
]





class LoRA_Left_Column_learn_Q_Module(nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
            self,
            lora_name,
            org_module: nn.Module,
            multiplier=1.0,
            lora_dim=4,
            alpha=1,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim

        if (org_module.weight.shape[0] < 40 * self.lora_dim):
            self.lora_dim = org_module.weight.shape[0]//40

        if "Linear" in org_module.__class__.__name__:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_up = nn.Linear(lora_dim, out_dim, bias=False)
            # self.QT = nn.Linear(in_dim, lora_dim)
            # self.Q = nn.Linear(lora_dim, out_dim)


        elif "Conv" in org_module.__class__.__name__:  # 一応
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels

            self.lora_dim = min(self.lora_dim, in_dim, out_dim)
            if self.lora_dim != lora_dim:
                print(f"{lora_name} dim (rank) is changed to: {self.lora_dim}")

            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = nn.Conv2d(
                in_dim, self.lora_dim, kernel_size, stride, padding, bias=False
            )
            self.lora_up = nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)
            # self.complementary = nn.Conv2d(
            #     self.lora_dim, out_dim, kernel_size, stride, padding, bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha #/ self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        # same as microsoft's
        nn.init.kaiming_uniform_(self.lora_down.weight, a=1)
        nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying

    def apply_to(self):
        # if (self.org_module.weight.shape[0] > 20*self.lora_dim):

        if (len(self.lora_down.weight.shape) == 2):
            Q, R = torch.linalg.qr(self.lora_up.weight[:, :].to('cuda:0'))
            # QQT = torch.mm(Q, Q.t()).to('cuda:0')
            # complementary_space_component = torch.einsum('ab,be->ae', QQT,
            #                                                   self.org_module.weight.to(torch.float32))
            # W_pivot = self.org_module.weight # - complementary_space_component
            # with torch.no_grad():
            # self.org_module.weight = torch.nn.Parameter(W_pivot.half())
            self.lora_down.weight = nn.Parameter(torch.mm(Q.t(), self.org_module.weight.float()))
            self.lora_up.weight = nn.Parameter(Q)
        if (len(self.lora_down.weight.shape) == 4):
            flattened_tensor = self.lora_up.weight.view(self.lora_up.weight.size(0), -1)
            Q, R = torch.linalg.qr(flattened_tensor.to('cuda:0'))
            flatten_W = self.org_module.weight.view(self.org_module.weight.size(0), -1).float()
            QTW = torch.mm(Q.t(), flatten_W).view(self.lora_down.weight.size())
            self.lora_down.weight = nn.Parameter(QTW)
            self.lora_up.weight = nn.Parameter(Q.view(self.lora_up.weight.size()))

        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        # print(x.shape, self.org_forward(x).shape, self.org_module.weight.shape, self.lora_up(self.lora_down(x)).shape)
        # return (
        #     self.org_forward(x)
        #     + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale
        # )
        # return (
        #         self.org_forward(x)
        # )
        return (
                self.org_forward(x)
                - self.lora_up(self.lora_down(x))
        )
        # return (
        #         self.org_forward(x) - self.complementary(x)
        # )


class LoRA_Left_Column_learn_Q_Network(nn.Module):
    def __init__(
            self,
            unet: UNet2DConditionModel,
            rank: int = 4,
            multiplier: float = 1.0,
            alpha: float = 1.0,
            train_method: TRAINING_METHODS = "full",
    ) -> None:
        super().__init__()
        self.lora_scale = 1
        self.multiplier = multiplier
        self.lora_dim = rank
        self.alpha = alpha

        # LoRAのみ
        self.module = LoRA_Left_Column_learn_Q_Module

        # unetのloraを作る
        self.unet_loras = self.create_modules(
            LORA_PREFIX_UNET,
            unet,
            DEFAULT_TARGET_REPLACE,
            self.lora_dim,
            self.multiplier,
            train_method=train_method,
        )
        print(f"create LoRA for U-Net: {len(self.unet_loras)} modules.")

        # assertion 名前の被りがないか確認しているようだ
        lora_names = set()
        for lora in self.unet_loras:
            assert (
                    lora.lora_name not in lora_names
            ), f"duplicated lora name: {lora.lora_name}. {lora_names}"
            lora_names.add(lora.lora_name)

        # 適用する
        for lora in self.unet_loras:
            lora.apply_to()
            self.add_module(
                lora.lora_name,
                lora,
            )

        del unet

        torch.cuda.empty_cache()

    def create_modules(
            self,
            prefix: str,
            root_module: nn.Module,
            target_replace_modules: List[str],
            rank: int,
            multiplier: float,
            train_method: TRAINING_METHODS,
    ) -> list:
        loras = []
        names = []
        for name, module in root_module.named_modules():
            if train_method == "noxattn" or train_method == "noxattn-hspace" or train_method == "noxattn-hspace-last":  # Cross Attention と Time Embed 以外学習
                if "attn2" in name or "time_embed" in name:
                    continue
            elif train_method == "innoxattn":  # Cross Attention 以外学習
                if "attn2" in name:
                    continue
            elif train_method == "selfattn":  # Self Attention のみ学習
                if "attn1" not in name:
                    continue
            elif train_method == "xattn" or train_method == "xattn-strict":  # Cross Attention のみ学習
                if "attn2" not in name:
                    continue
            elif train_method == "full":  # 全部学習
                pass
            else:
                raise NotImplementedError(
                    f"train_method: {train_method} is not implemented."
                )
            if module.__class__.__name__ in target_replace_modules:
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ in ["Linear", "Conv2d", "LoRACompatibleLinear",
                                                           "LoRACompatibleConv"]:
                        if train_method == 'xattn-strict':
                            if 'out' in child_name:
                                continue
                        if train_method == 'noxattn-hspace':
                            if 'mid_block' not in name:
                                continue
                        if train_method == 'noxattn-hspace-last':
                            if 'mid_block' not in name or '.1' not in name or 'conv2' not in child_name:
                                continue
                        lora_name = prefix + "." + name + "." + child_name
                        lora_name = lora_name.replace(".", "_")
                        # print(f"{lora_name}")
                        lora = self.module(
                            lora_name, child_module, multiplier, rank, self.alpha
                        )
                        #                         print(name, child_name)
                        #                         print(child_module.weight.shape)
                        if lora_name not in names:
                            loras.append(lora)
                            names.append(lora_name)
        #         print(f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@ \n {names}')
        print(f"loras list length is {len(loras)}")
        print(f"names list length is {len(names)}")
        return loras

    def prepare_optimizer_params(self):
        all_params = []

        if self.unet_loras:  # 実質これしかない
            params = []
            [params.extend(lora.parameters()) for lora in self.unet_loras]
            param_data = {"params": params}
            all_params.append(param_data)

        return all_params

    def save_weights(self, file, dtype=None, metadata: Optional[dict] = None):
        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        #         for key in list(state_dict.keys()):
        #             if not key.startswith("lora"):
        #                 # lora以外除外
        #                 del state_dict[key]

        if os.path.splitext(file)[1] == ".safetensors":
            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def set_lora_slider(self, scale):
        self.lora_scale = scale

    def __enter__(self):
        for lora in self.unet_loras:
            lora.multiplier = 1.0 * self.lora_scale

    def __exit__(self, exc_type, exc_value, tb):
        for lora in self.unet_loras:
            lora.multiplier = 0
            
            
            
            
            
            
            
prompts_main_and_accessory = ["A {} wearing a {}, standing at the edge of a fountain in a city park, with water splashing around"]

prompts_main_and_trivial = ["An {} is trying to open a {}  with its paws, located in a nostalgic kitchen filled with vintage furniture and scattered biscuit",
                           "A {} wearing a blue apron is tapping a giant {} with a miniature hammer standing on a wooden workbench cluttered with tools and covered with a red handcraft tool cloth"]



main_objects = {
    'bear_plushie': 'bear <lora1.0:bear>',
    'cat': 'cat <lora1.0:cat0>',
    'cat2': 'cat <lora1.0:cat2>',
    'dog': 'dog <lora1.0:Corgi>',
    'dog2': 'dog <lora1.0:chowchow>',
    'dog3': 'dog <lora1.0:dog3>',
    'dog5': 'dog <lora1.0:Dachshund>',
    'dog6': 'dog <lora1.0:dog6>',
    'dog7': 'dog <lora1.0:dog7>',
    'dog8': 'dog <lora1.0:BorderCollie>',
    'ducktoy': 'duck <lora1.0:ducktoy>',
    'grey_sloth_plushie': 'sloth <lora1.0:grey_sloth_plushie>',
    'monster_toy': 'toy <lora1.0:monster_toy>',
    'red_cartoon': 'monster <lora1.0:red_cartoon>',
    'robot_toy': 'toy <lora1.0:robot_toy>',
    'wolf_plushie': 'wolf <lora1.0:wolf_plushie>'
}

accessory_objects = {
    'backpackdog':'backpack <lora1.0:backpack_dog>',
    'fancyboot': 'boot <lora1.0:fancyboot>',
    'pink_sunglasses': 'Sunglasses <lora1.0:pink_sunglasses>',
    'shiny_sneaker': 'shoe <lora1.0:shiny_sneaker>',
}


trivial_objects = {
    'berry_bowl': 'bowl <lora1.0:berry_bowl>',
    'can': 'can <lora1.0:Can>',
    'candle': 'candle <lora1.0:candle>',
    'clock': 'clock <lora1.0:Clock>',
    'poop_emoji': 'toy <lora1.0:poop_emoji>',
    'rc_car': 'toy car <lora1.0:rc_car>',
    'teapot': 'teapot <lora1.0:teapot>',
    'vase': 'vase <lora1.0:vase>'
}



ranks_trained = [1, 2, 4, 8, 16, 32, 64]
models = ['left_column_learn_Q', 'lora']





def combine_sample_plot(tuning_model_name, prompt, object1, rank_of_model1, object2, rank_of_model2):
    path1 = f'models/{object1}XL_{tuning_model_name}_Linear_and_Conv2_alpha1.0_rank{rank_of_model1}_noxattn/{object1}XL_{tuning_model_name}_Linear_and_Conv2_alpha1.0_rank{rank_of_model1}_noxattn_last.pt'
    path2 = f'models/{object2}XL_{tuning_model_name}_Linear_and_Conv2_alpha1.0_rank{rank_of_model2}_noxattn/{object2}XL_{tuning_model_name}_Linear_and_Conv2_alpha1.0_rank{rank_of_model2}_noxattn_last.pt'
    lora_weights = [path1, path2]
    main_folder = f'combine_image_{tuning_model_name}'
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)
    
    subfolder_path = os.path.join(main_folder, f'{object1}_and_{object2}')
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)


    image_prompt_dir_name = '_'.join(prompt.replace(',', ' ').split())
    image_prompt_dir = os.path.join(subfolder_path, image_prompt_dir_name)
    if not os.path.exists(image_prompt_dir):
        os.makedirs(image_prompt_dir)
    
    
    
    seeds = [8685, 5555, 4000, 4555]
    weight_dtype = torch.float16
    start_noise = 700
    num_images_per_prompt = 1

    scales = [1]
    

    def build_lora_from_weight(lora_weight, unet):

        if 'full' in lora_weight:
            train_method = 'full'
        elif 'noxattn' in lora_weight:
            train_method = 'noxattn'
        else:
            train_method = 'noxattn'
        network_type = "c3lier"
        if train_method == 'xattn':
            network_type = 'lierla'
        modules = DEFAULT_TARGET_REPLACE
        if network_type == "c3lier":
            modules += UNET_TARGET_REPLACE_MODULE_CONV
        model_name = lora_weight
        name = os.path.basename(model_name)
        rank = 4
        alpha = 1
        rankstr = re.findall(r'rank(\d+)', lora_weight)
        rank = int(rankstr[0])
        alphastr = re.findall(r'alpha(\d+)', lora_weight)
        alpha = float(alphastr[0])
        network_sub = LoRA_Left_Column_learn_Q_Network(
                unet,
                rank=rank,
                multiplier=1.0,
                alpha=alpha,
                train_method=train_method,
            ).to(device, dtype=weight_dtype)
        if ('safetensors' in lora_weight):
            load_file(network_sub, lora_weight)
        else:
            network_sub.load_state_dict(torch.load(lora_weight))
        return network_sub
    
    pipe = StableDiffusionXLPipeline.from_single_file("stable-diffusion-xl-base-1.0/sd_xl_base_1.0.safetensors", torch_dtype=weight_dtype)
    pipe.__call__ = call
    pipe = pipe.to(device)
    unet = pipe.unet
    print(f"{lora_weights[0]}\n{lora_weights[1]}")
    network1 = build_lora_from_weight(lora_weights[0], unet)
    network2 = build_lora_from_weight(lora_weights[1], unet)
    
    
    if 'full' in lora_weights[0]:
        train_method = 'full'
    elif 'noxattn' in lora_weights[0]:
        train_method = 'noxattn'
    else:
        train_method = 'noxattn'
    network_type = "c3lier"
    if train_method == 'xattn':
        network_type = 'lierla'
    modules = DEFAULT_TARGET_REPLACE
    if network_type == "c3lier":
        modules += UNET_TARGET_REPLACE_MODULE_CONV
    model_name = lora_weights[0]
    name = os.path.basename(model_name)
    rank = 4
    alpha = 1
    rankstr1 = re.findall(r'rank(\d+)', lora_weights[0])
    rank1 = int(rankstr1[0])
    rankstr2 = re.findall(r'rank(\d+)', lora_weights[1])
    rank2 = int(rankstr2[0])
    rank = rank1 + rank2
    alphastr = re.findall(r'alpha(\d+)', lora_weights[0])
    alpha = float(alphastr[0])
    
    
    
    for _ in range(num_images_per_prompt):
        image_list = []

        unet = pipe.unet
        for scale in scales:
            for seed in seeds:
                generator = torch.manual_seed(seed)
                images = pipe(prompt, num_images_per_prompt=1, num_inference_steps=50, generator=generator, network=network2, start_noise=start_noise, scale=scale, unet=unet).images[0]
                image_list.append(images)
        # del unet, network, pipe
        del unet, pipe
        unet = None
        network = None
        pipe = None
        torch.cuda.empty_cache()
        flush()

        for idx, image in enumerate(image_list):
            for i in range(len(scales)):
                save_path = os.path.join(image_prompt_dir, f'image_{seeds[idx]}_rank1_{rank1}_rank2_{rank2}.png')
                image.save(save_path)
    
            




# object = list(main_objects.keys())[0]
# tuning_model_name = models[0]
# rank_of_model = ranks_trained[4]
# prompt = "A cat <lora1.0:cat0> wearing a sunglasses <lora1.0:pink_sunglasses>, standing at the edge of a fountain in a city park, with water splashing around"



tuning_model_name = models[0]
for i_main_o, main_object in enumerate(list(main_objects.keys())):
    object1 = list(main_objects.keys())[i_main_o]
    for j_trivial_o, trivial_object in enumerate(list(trivial_objects.keys())):
        object2 = list(trivial_objects.keys())[j_trivial_o]
        prompt = prompts_main_and_trivial[0].format(list(main_objects.values())[i_main_o], list(trivial_objects.values())[j_trivial_o])
        for rank1 in ranks_trained:
            for rank2 in ranks_trained:
                combine_sample_plot(tuning_model_name, prompt, object1, rank1, object2, rank2)


                    
