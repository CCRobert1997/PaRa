# ref:
# - https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py
# - https://github.com/kohya-ss/sd-scripts/blob/main/networks/lora.py

import os
import math
from typing import Optional, List, Type, Set, Literal

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from safetensors.torch import save_file


UNET_TARGET_REPLACE_MODULE_TRANSFORMER = [
#     "Transformer2DModel",  # どうやらこっちの方らしい？ # attn1, 2
    "Attention"
]
UNET_TARGET_REPLACE_MODULE_CONV = [
    "ResnetBlock2D",
    "Downsample2D",
    "Upsample2D",
    #     "DownBlock2D",
    #     "UpBlock2D"
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


class LoRAModule(nn.Module):
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

        if "Linear" in org_module.__class__.__name__:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_up = nn.Linear(lora_dim, out_dim, bias=False)

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

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        # same as microsoft's
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        return (
            self.org_forward(x)
            + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale
        )


class LoRANetwork(nn.Module):
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
        self.module = LoRAModule

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
                    if child_module.__class__.__name__ in ["Linear", "Conv2d", "LoRACompatibleLinear", "LoRACompatibleConv"]:
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
#                         print(f"{lora_name}")
                        lora = self.module(
                            lora_name, child_module, multiplier, rank, self.alpha
                        )
#                         print(name, child_name)
#                         print(child_module.weight.shape)
                        if lora_name not in names:
                            loras.append(lora)
                            names.append(lora_name)
#         print(f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@ \n {names}')
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





class LoRA_Light_Left_Column_learn_Q_Module(nn.Module):
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
            self.lora_up = nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)
            # self.complementary = nn.Conv2d(
            #     self.lora_dim, out_dim, kernel_size, stride, padding, bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha #/ self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        # same as microsoft's
        # nn.init.kaiming_uniform_(self.lora_down.weight, a=1)
        nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying

    def apply_to(self):
        # if (self.org_module.weight.shape[0] > 20*self.lora_dim):
        if "Linear" in self.org_module.__class__.__name__:
            Q, R = torch.linalg.qr(self.lora_up.weight[:, :].to('cuda:0'))
            # QQT = torch.mm(Q, Q.t()).to('cuda:0')
            # complementary_space_component = torch.einsum('ab,be->ae', QQT,
            #                                                   self.org_module.weight.to(torch.float32))
            # W_pivot = self.org_module.weight # - complementary_space_component
            # with torch.no_grad():
            # self.org_module.weight = torch.nn.Parameter(W_pivot.half())
            self.lora_down_new = nn.Linear(self.org_module.in_features, self.lora_dim, bias=False)
            self.lora_down_new.weight = nn.Parameter(torch.mm(Q.t(), self.org_module.weight.float()))
            self.lora_up.weight = nn.Parameter(Q)
        if "Conv" in self.org_module.__class__.__name__:
            self.lora_down_new = nn.Conv2d(
                self.org_module.in_channels, self.lora_dim, self.org_module.kernel_size, self.org_module.stride, self.org_module.padding, bias=False
            )
            flattened_tensor = self.lora_up.weight.view(self.lora_up.weight.size(0), -1)
            Q, R = torch.linalg.qr(flattened_tensor.to('cuda:0'))
            flatten_W = self.org_module.weight.view(self.org_module.weight.size(0), -1).float()
            QTW = torch.mm(Q.t(), flatten_W).view(self.lora_down_new.weight.size())
            self.lora_down_new.weight = nn.Parameter(QTW)
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
        return (
                self.org_forward(x)
                - self.lora_up(self.lora_down_new(x))
        )
        # return (
        #         self.org_forward(x) - self.complementary(x)
        # )


class LoRA_Light_Left_Column_learn_Q_Network(nn.Module):
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
        self.module = LoRA_Light_Left_Column_learn_Q_Module

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
        # state_dict = self.state_dict()
        # for name in state_dict.keys():
        #     print(name)
        full_state_dict = self.state_dict()

        # Create a new dictionary excluding keys that contain 'lora_down_new'
        state_dict = {key: value for key, value in full_state_dict.items() if "lora_down_new" not in key}


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


class LoRAControllerModule(nn.Module):
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

        if "Linear" in org_module.__class__.__name__:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_up = nn.Linear(lora_dim, out_dim, bias=False)

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

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        # self.scale = alpha / self.lora_dim
        self.scale = alpha
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        # same as microsoft's
        nn.init.kaiming_uniform_(self.lora_down.weight, a=1)
        nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying

    def apply_to(self):
        # print(self.org_module.weight.shape, self.lora_down.weight.shape, self.lora_up.weight.shape)
        # lora_up_weight_norm = self.lora_down.weight.reshape(self.lora_down.weight.size(0), -1).norm(dim=1, keepdim=True).reshape(self.lora_down.weight.size(0), 1, 1, 1)
        # lora_up_normalized_weight = self.lora_down.weight / lora_up_weight_norm
        # dot_w0A = torch.einsum('abcd,ebcd->ae', self.org_module.weight.to(torch.float32), lora_up_normalized_weight.to('cuda:0'))
        # print(self.lora_up.weight.shape, self.lora_down.weight.shape)
        # ?????????????this is work on A, maybe not good????????????????????????
        if (len(self.lora_down.weight.shape) == 4):
            d, x_input_n, channel_1, channel_2 = self.lora_down.weight.shape
            lora_down_t = self.lora_down.weight.permute(1, 0, 2, 3)
            Q, R = torch.linalg.qr(lora_down_t[:, :, 0, 0])
            combinedQR = [torch.linalg.qr(lora_down_t[:, :, i, j])[0] for i in range(lora_down_t.shape[2]) for j in
                          range(lora_down_t.shape[3])]
            Q_stacked = torch.stack(combinedQR, dim=0)
            Q_final = Q_stacked.view(channel_1, channel_2, x_input_n, d).permute(2, 3, 0, 1).to(
                'cuda:0')  # [320, 4, 3, 3]
            # self.org_module.weight [320, 320, 3, 3]
            dot_w0A = torch.einsum('abcd,becd->aecd', self.org_module.weight.to(torch.float32),
                                   Q_final)  # [320, 4, 3, 3]

            complementary_space_component = torch.einsum('abcd,ebcd->aecd', dot_w0A, Q_final)
            W_pivot = self.org_module.weight - complementary_space_component
            # print(W_pivot.shape)
            with torch.no_grad():
                self.org_module.weight = torch.nn.Parameter(W_pivot.to(torch.bfloat16))# .half())
        else:
            lora_down_t = self.lora_down.weight.permute(1, 0)
            Q, R = torch.linalg.qr(lora_down_t[:, :])
            Q_final = Q.to('cuda:0')  # [1280, 4]
            # self.org_module.weight [320, 1280]
            dot_w0A = torch.einsum('ab,bc->ac', self.org_module.weight.to(torch.float32), Q_final)  # [320, 4]

            complementary_space_component = torch.einsum('ab,cb->ac', dot_w0A, Q_final)
            W_pivot = self.org_module.weight - complementary_space_component
            # print(W_pivot.shape)
            with torch.no_grad():
                self.org_module.weight = torch.nn.Parameter(W_pivot.to(torch.bfloat16))# .half())

        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        # print(x.shape, self.org_forward(x).shape, self.org_module.weight.shape, self.lora_up(self.lora_down(x)).shape)
        # return (
        #     self.org_forward(x)
        #     + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale
        # )
        return (
            self.org_forward(x)
        )


class LoRAControllerNetwork(nn.Module):
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
        self.module = LoRAControllerModule

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
                        #                         print(f"{lora_name}")
                        lora = self.module(
                            lora_name, child_module, multiplier, rank, self.alpha
                        )
                        #                         print(name, child_name)
                        #                         print(child_module.weight.shape)
                        if lora_name not in names:
                            loras.append(lora)
                            names.append(lora_name)
        #         print(f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@ \n {names}')
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


class LoRA_Left_Column_ControllerModule(nn.Module):
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

        if "Linear" in org_module.__class__.__name__:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_up = nn.Linear(lora_dim, out_dim, bias=False)
            # self.complementary = nn.Linear(lora_dim, lora_dim)


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
            # self.complementary = nn.Conv2d(self.lora_dim, self.lora_dim, kernel_size, stride, padding, bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        # same as microsoft's
        nn.init.kaiming_uniform_(self.lora_down.weight, a=1)
        nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying

    def apply_to(self):

        if (len(self.lora_down.weight.shape) == 4):
            # d, x_input_n, channel_1, channel_2 = self.lora_up.weight.shape
            Q, R = torch.linalg.qr(self.lora_up.weight[:, :, 0, 0])
            # print(torch.mm(torch.mm(Q, Q.t()).to('cuda:0'), self.org_module.weight))
            QQT = torch.mm(Q, Q.t()).to('cuda:0')
            complementary_space_component = torch.einsum('ab,becd->aecd', QQT,
                                                              self.org_module.weight.to(torch.float32))
            W_pivot = self.org_module.weight - complementary_space_component
            # with torch.no_grad():
            self.org_module.weight = torch.nn.Parameter(W_pivot.to(torch.bfloat16))#.half())
            # self.complementary.weight = nn.Parameter(complementary_space_component)
        else:
            Q, R = torch.linalg.qr(self.lora_up.weight[:, :])
            QQT = torch.mm(Q, Q.t()).to('cuda:0')
            complementary_space_component = torch.einsum('ab,be->ae', QQT,
                                                              self.org_module.weight.to(torch.float32))
            W_pivot = self.org_module.weight # - complementary_space_component
            # with torch.no_grad():
            self.org_module.weight = torch.nn.Parameter(W_pivot.to(torch.bfloat16))#.half())
            # self.complementary.weight = nn.Parameter(complementary_space_component)

        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        # print(x.shape, self.org_forward(x).shape, self.org_module.weight.shape, self.lora_up(self.lora_down(x)).shape)
        # return (
        #     self.org_forward(x)
        #     + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale
        # )
        return (
                self.org_forward(x)
        )


class LoRA_Left_Column_ControllerNetwork(nn.Module):
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
        self.module = LoRA_Left_Column_ControllerModule

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
                        #                         print(f"{lora_name}")
                        lora = self.module(
                            lora_name, child_module, multiplier, rank, self.alpha
                        )
                        #                         print(name, child_name)
                        #                         print(child_module.weight.shape)
                        if lora_name not in names:
                            loras.append(lora)
                            names.append(lora_name)
        #         print(f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@ \n {names}')
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
            lora_ratio=40
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim

        if (org_module.weight.shape[0] < lora_ratio * self.lora_dim):
            self.lora_dim = org_module.weight.shape[0]//lora_ratio

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
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        # same as microsoft's
        nn.init.kaiming_uniform_(self.lora_down.weight, a=1)
        nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying

    def apply_to(self):
        # if (self.org_module.weight.shape[0] > 20 * self.lora_dim):
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
            # self.complementary.weight = nn.Parameter(Q)
        if (len(self.lora_down.weight.shape) == 4):
            # print(self.lora_up.weight.shape)
            # print(self.lora_down.weight.shape)
            # print(self.org_module.weight.shape)
            flattened_tensor = self.lora_up.weight.view(self.lora_up.weight.size(0), -1)
            # print(f'conv shape {flattened_tensor.shape}')
            Q, R = torch.linalg.qr(flattened_tensor.to('cuda:0'))
            # print(Q.shape)
            flatten_W = self.org_module.weight.view(self.org_module.weight.size(0), -1).float()
            # print(flatten_W.shape)
            QTW = torch.mm(Q.t(), flatten_W).view(self.lora_down.weight.size())
            self.lora_down.weight = nn.Parameter(QTW)
            self.lora_up.weight = nn.Parameter(Q.view(self.lora_up.weight.size()))
            # print("-----------------------")

        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        # print(x.shape, self.org_forward(x).shape, self.org_module.weight.shape, self.lora_up(self.lora_down(x)).shape)
        # return (
        #     self.org_forward(x)
        #     + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale
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
            lora_ratio: int = 40,
    ) -> None:
        super().__init__()
        self.lora_scale = 1
        self.multiplier = multiplier
        self.lora_dim = rank
        self.alpha = alpha
        self.lora_ratio = lora_ratio

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
                        #                         print(f"{lora_name}")
                        lora = self.module(
                            lora_name, child_module, multiplier, rank, self.alpha, self.lora_ratio
                        )
                        #                         print(name, child_name)
                        #                         print(child_module.weight.shape)
                        if lora_name not in names:
                            loras.append(lora)
                            names.append(lora_name)
        #         print(f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@ \n {names}')
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