import torch
import copy
from typing import List, Tuple
import numpy as np
import torch.nn as nn
import math
import torch.nn.functional as F
import utils.tf_spatial_transform_local as tf_spatial_transform_local
import utils.torch_tps_transform as torch_tps_transform
import utils.constant as constant

grid_w = constant.GRID_W
grid_h = constant.GRID_H
device = constant.GPU_DEVICE


def shift2mesh(mesh_shift, height,width):
    device = mesh_shift.device
    batch_size = mesh_shift.shape[0]
    h = height / grid_h
    w = width / grid_w
    # print(width,height,grid_w,grid_h)
    w_list = torch.arange(0.,height+1.,w)
    w_list_arr = w_list.repeat((grid_h+1))
    h_list = torch.arange(0., height + 1., h)
    h_list_arr = h_list.repeat((grid_w+1),1)
    h_list_arr = h_list_arr.T
    h_list_arr = h_list_arr.contiguous().view(-1)
    # print(w_list_arr,h_list_arr)
    ori_pt = torch.stack((w_list_arr,h_list_arr),dim=1).to(device)#.cuda(non_blocking=True)
    ori_pt = ori_pt.view(grid_h + 1, grid_w + 1, 2)
    # print(ori_pt)
    ori_pt = torch.tile(ori_pt.unsqueeze(0), [batch_size, 1, 1, 1])
    ori_pt = ori_pt.to(device)#.cuda(non_blocking=True)
    # print("ori_pt:",ori_pt.shape)
    # print("mesh_shift:", mesh_shift.shape)
    tar_pt = ori_pt + mesh_shift
    return tar_pt

def shift2mesh0(mesh_shift, height,width):
    device = mesh_shift.device
    batch_size = mesh_shift.shape[0]
    h = height / grid_h
    w = width / grid_w
    ori_pt = []
    for i in range(grid_h + 1):
        for j in range(grid_w + 1):
            ww = j * w
            hh = i * h
            p = torch.FloatTensor([ww, hh])
            ori_pt.append(p.unsqueeze(0))
    ori_pt = torch.cat(ori_pt,dim=0)
    # print(ori_pt.shape)
    ori_pt = ori_pt.view(grid_h + 1, grid_w + 1, 2)
    # print(ori_pt)
    ori_pt = torch.tile(ori_pt.unsqueeze(0), [batch_size, 1, 1, 1])
    ori_pt = ori_pt.to(device)#.cuda(non_blocking=True)
    # print("ori_pt:",ori_pt.shape)
    # print("mesh_shift:", mesh_shift.shape)
    tar_pt = ori_pt + mesh_shift
    return tar_pt

def get_rigid_mesh(batch_size, height, width):

    ww = torch.matmul(torch.ones([grid_h+1, 1]), torch.unsqueeze(torch.linspace(0., float(width), grid_w+1), 0))
    hh = torch.matmul(torch.unsqueeze(torch.linspace(0.0, float(height), grid_h+1), 1), torch.ones([1, grid_w+1]))
    ww = ww.to(device)#.cuda(non_blocking=True)
    hh = hh.to(device)#.cuda(non_blocking=True)

    ori_pt = torch.cat((ww.unsqueeze(2), hh.unsqueeze(2)),2) # (grid_h+1)*(grid_w+1)*2
    ori_pt = ori_pt.unsqueeze(0).expand(batch_size, -1, -1, -1)

    return ori_pt

def get_norm_mesh(mesh, height, width):
    batch_size = mesh.size()[0]
    mesh_w = mesh[...,0]*2./float(width) - 1.
    mesh_h = mesh[...,1]*2./float(height) - 1.
    norm_mesh = torch.stack([mesh_w, mesh_h], 3) # bs*(grid_h+1)*(grid_w+1)*2
    # norm_mesh = torch.stack([mesh_h, mesh_w], 3)  # bs*(grid_h+1)*(grid_w+1)*2
    # print("norm_mesh:",norm_mesh.shape)
    return norm_mesh.reshape([batch_size, -1, 2]) # bs*-1*2

def H2Mesh(H, rigid_mesh):

    H_inv = torch.inverse(H)
    ori_pt = rigid_mesh.reshape(rigid_mesh.size()[0], -1, 2)
    ones = torch.ones(rigid_mesh.size()[0], (grid_h+1)*(grid_w+1),1)
    ori_pt = ori_pt.to(device)#.cuda(non_blocking=True)
    ones = ones.to(device)#.cuda(non_blocking=True)

    ori_pt = torch.cat((ori_pt, ones), 2) # bs*(grid_h+1)*(grid_w+1)*3
    tar_pt = torch.matmul(H_inv, ori_pt.permute(0,2,1)) # bs*3*(grid_h+1)*(grid_w+1)

    mesh_x = torch.unsqueeze(tar_pt[:,0,:]/tar_pt[:,2,:], 2)
    mesh_y = torch.unsqueeze(tar_pt[:,1,:]/tar_pt[:,2,:], 2)
    mesh = torch.cat((mesh_x, mesh_y), 2).reshape([rigid_mesh.size()[0], grid_h+1, grid_w+1, 2])

    return mesh

class RepBlock(nn.Module):
    """
    MobileOne-style residual blocks, including residual joins and re-parameterization convolutions
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 1,
        inference_mode: bool = False,
        rbr_conv_kernel_list: List[int] = [7, 3],
        use_bn_conv: bool = False,
        act_layer: nn.Module = nn.ReLU,
        skip_include_bn: bool = True,
    ) -> None:
        """Construct a Re-parameterization module.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param stride: Stride for convolution.
        :param groups: Number of groups for convolution.
        :param inference_mode: Whether to use inference mode.
        :param rbr_conv_kernel_list: List of kernel sizes for re-parameterizable convolutions.
        :param use_bn_conv: Whether the bn is in front of conv, if false, conv is in front of bn
        :param act_layer: Activation layer.
        :param skip_include_bn: Whether to include bn in skip connection.
        """
        super(RepBlock, self).__init__()

        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rbr_conv_kernel_list = sorted(rbr_conv_kernel_list, reverse=True)
        self.num_conv_branches = len(self.rbr_conv_kernel_list)
        self.kernel_size = self.rbr_conv_kernel_list[0]
        self.use_bn_conv = use_bn_conv
        self.skip_include_bn = skip_include_bn

        self.activation = act_layer()

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=stride,
                padding=self.kernel_size // 2,
                groups=groups,
                bias=True,
            )
        else:
            # Re-parameterizable skip connection
            if out_channels == in_channels and stride == 1:
                if self.skip_include_bn:
                    # Use residual connections that include BN
                    self.rbr_skip = nn.BatchNorm2d(num_features=in_channels)
                else:
                    # Use residual connections
                    self.rbr_skip = nn.Identity()
            else:
                # Use residual connections
                self.rbr_skip = None

            # Re-parameterizable conv branches
            rbr_conv = list()
            for kernel_size in self.rbr_conv_kernel_list:
                if self.use_bn_conv:
                    rbr_conv.append(
                        self._bn_conv(
                            in_chans=in_channels,
                            out_chans=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=kernel_size // 2,
                            groups=groups,
                        )
                    )
                else:
                    rbr_conv.append(
                        self._conv_bn(
                            in_chans=in_channels,
                            out_chans=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=kernel_size // 2,
                            groups=groups,
                        )
                    )

            self.rbr_conv = nn.ModuleList(rbr_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward pass."""
        # Inference mode forward pass.
        if self.inference_mode:
            return self.activation(self.reparam_conv(x))
        # print("not inference_mode")
        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        # Other branches
        out = identity_out
        for ix in range(self.num_conv_branches):
            out = out + self.rbr_conv[ix](x)
        return self.activation(out)

    def reparameterize(self):
        """Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(
            in_channels=self.rbr_conv[0].conv.in_channels,
            out_channels=self.rbr_conv[0].conv.out_channels,
            kernel_size=self.rbr_conv[0].conv.kernel_size,
            stride=self.rbr_conv[0].conv.stride,
            padding=self.rbr_conv[0].conv.padding,
            dilation=self.rbr_conv[0].conv.dilation,
            groups=self.rbr_conv[0].conv.groups,
            bias=True,
        )
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # Delete un-used branches
        for para in self.parameters():
            para.detach_()
        self.__delattr__("rbr_conv")
        if hasattr(self, "rbr_skip"):
            self.__delattr__("rbr_skip")

        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        :return: Tuple of (kernel, bias) after fusing branches.
        """

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_skip_tensor(self.rbr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            if self.use_bn_conv:
                _kernel, _bias = self._fuse_bn_conv_tensor(self.rbr_conv[ix])
            else:
                _kernel, _bias = self._fuse_conv_bn_tensor(self.rbr_conv[ix])
            # pad kernel
            if _kernel.shape[-1] < self.kernel_size:
                pad = (self.kernel_size - _kernel.shape[-1]) // 2
                _kernel = torch.nn.functional.pad(_kernel, [pad, pad, pad, pad])

            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_identity
        bias_final = bias_conv + bias_identity
        return kernel_final, bias_final

    def _fuse_skip_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param branch: skip branch, maybe include bn layer
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """

        if not hasattr(self, "id_tensor"):
            input_dim = self.in_channels // self.groups
            kernel_value = torch.zeros(
                (self.in_channels, input_dim, self.kernel_size, self.kernel_size),
                dtype=self.rbr_conv[0].conv.weight.dtype,
                device=self.rbr_conv[0].conv.weight.device,
            )
            for i in range(self.in_channels):
                kernel_value[
                    i, i % input_dim, self.kernel_size // 2, self.kernel_size // 2
                ] = 1
            self.id_tensor = kernel_value
        if isinstance(branch, nn.Identity):
            kernel = self.id_tensor
            return kernel, torch.zeros(
                (self.in_channels),
                dtype=self.rbr_conv[0].conv.weight.dtype,
                device=self.rbr_conv[0].conv.weight.device,
            )
        else:
            assert isinstance(
                branch, nn.BatchNorm2d
            ), "Make sure the module in skip is nn. BatchNorm2d"
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std

    def _fuse_bn_conv_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """先bn,后conv

        :param branch:
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """
        kernel = branch.conv.weight
        running_mean = branch.bn.running_mean
        running_var = branch.bn.running_var
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn.eps
        std = (running_var + eps).sqrt()
        t = gamma / std
        t = torch.stack([t] * (kernel.shape[0] * kernel.shape[1]//t.shape[0]),dim=0).reshape(-1, self.in_channels // self.groups, 1, 1)
        t_beta = torch.stack([beta] * (kernel.shape[0] * kernel.shape[1]//beta.shape[0]),dim=0).reshape(-1, self.in_channels // self.groups, 1, 1)
        t_running_mean = torch.stack([running_mean] * (kernel.shape[0] * kernel.shape[1]//running_mean.shape[0]),dim=0).reshape(-1, self.in_channels // self.groups, 1, 1)
        return kernel * t, torch.sum(
            kernel
            * (
                t_beta - t_running_mean * t
            ),
            dim=(1, 2, 3),
        )

    def _fuse_conv_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """First conv, then bn

        :param branch:
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """

        kernel = branch.conv.weight
        running_mean = branch.bn.running_mean
        running_var = branch.bn.running_var
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(
        self,
        in_chans: int,
        out_chans: int,
        kernel_size: int,
        stride: int,
        padding: int,
        groups: int,
    ) -> nn.Sequential:
        """First conv, then bn

        :param kernel_size: Size of the convolution kernel.
        :param padding: Zero-padding size.
        :return: Conv-BN module.
        """
        mod_list = nn.Sequential()
        mod_list.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_chans,
                out_channels=out_chans,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
        )
        mod_list.add_module("bn", nn.BatchNorm2d(num_features=out_chans))
        return mod_list

    def _bn_conv(
        self,
        in_chans: int,
        out_chans: int,
        kernel_size: int,
        stride: int,
        padding: int,
        groups: int,
    ) -> nn.Sequential:
        """Add bn first, then conv"""
        mod_list = nn.Sequential()
        mod_list.add_module("bn", nn.BatchNorm2d(num_features=in_chans))
        mod_list.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_chans,
                out_channels=out_chans,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
        )
        return mod_list


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_head_dim=None,
        gamma = 1/64
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim
        self.gamma = math.log(gamma)

        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # D = self._get_D(x)
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class AttnTokenMixer(nn.Module):
    def __init__(
        self,
        in_chans: int,
        num_heads: int,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_head_dim=None,
        inference_mode: bool = False,
        use_CPE: bool = False,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_chans)
        # self.norm = nn.GroupNorm(num_heads,in_chans)
        self.attn = Attention(
            in_chans,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attn_head_dim=attn_head_dim,
        )
        # self.norm = nn.LayerNorm(in_chans)
        if use_CPE:
            self.CPE = RepBlock(
                in_channels=in_chans,
                out_channels=in_chans,
                rbr_conv_kernel_list=[3],
                stride=1,
                groups=in_chans,
                inference_mode=inference_mode,
                act_layer=nn.Identity,
                skip_include_bn=False,
            )
        else:
            self.CPE = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, Hp, Wp = x.shape
        x = self.CPE(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.attn(self.norm(x))
        x = x.permute(0, 2, 1).reshape(B, C, Hp, Wp).contiguous()
        return x



class FastViTBlock(nn.Module):
    def __init__(
        self,
        chans: int,
        num_heads: int,
        inference_mode: bool,
        act_layer: nn.Module,
        use_attn: bool,
        expand_ratio: int,
        use_CPE: bool = False,
    ) -> None:
        super().__init__()
        if use_attn:
            # AttnTokenMixer
            self.token_mixer = AttnTokenMixer(
                chans, num_heads, inference_mode=inference_mode, use_CPE=use_CPE
            )
        else:
            # RepMixer
            self.token_mixer = RepBlock(
                in_channels=chans,
                out_channels=chans,
                stride=1,
                groups=chans,
                rbr_conv_kernel_list=[3],
                use_bn_conv=True,
                act_layer=act_layer,
                inference_mode=inference_mode,
                skip_include_bn=False,
            )
        mid_chans = chans * expand_ratio
        # conv_ffn
        self.conv_ffn = nn.Sequential(
            nn.Conv2d(chans, chans, kernel_size=7, padding=3, groups=chans, bias=False),
            nn.BatchNorm2d(chans),
            nn.Conv2d(chans, mid_chans, kernel_size=1, padding=0),
            act_layer(),
            nn.Conv2d(mid_chans, chans, kernel_size=1, padding=0),
        )
        # self.conv_ffn = nn.Sequential(
        #     nn.Conv2d(chans, mid_chans, kernel_size=7, padding=3, groups=chans, bias=False),
        #     # nn.BatchNorm2d(chans),
        #     # nn.Conv2d(chans, mid_chans, kernel_size=1, padding=0),
        #     nn.GroupNorm(8,mid_chans),
        #     act_layer(),
        #     nn.Conv2d(mid_chans, chans, kernel_size=1, padding=0),
        # )


    def forward(self, x):
        x = self.token_mixer(x)
        x = x + self.conv_ffn(x)
        return x


class FastStage(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        use_attn: bool,
        num_heads: int,
        num_blocks_per_stage: int,
        inference_mode: bool,
        act_layer: nn.Module,
        expand_ratio: int,
        use_patch_embed: bool,
    ) -> None:
        """
        Constructs a FastStage

        :param in_chans: Number of input channels.
        :param out_chans: Number of output channels.
        :param num_heads: Number of heads for attention If use_attn is True.
        :param use_attn: Whether to use attention.
        :param num_blocks_per_stage: Number of blocks per stage.
        :param inference_mode: Whether to use inference mode.
        :param act_layer: Activation layer.
        :param expand_ratio: Expansion ratio in conv_ffn.
        :param use_patch_embed: Whether to use patch embedding.
        """
        super().__init__()
        self.num_blocks_per_stage = num_blocks_per_stage
        if use_patch_embed:
            self.patch_embed = nn.Sequential(
                RepBlock(
                    in_channels=in_chans,
                    out_channels=in_chans,
                    rbr_conv_kernel_list=[7, 3],
                    stride=2,
                    groups=in_chans,
                    inference_mode=inference_mode,
                    act_layer=act_layer,
                ),
                RepBlock(
                    in_channels=in_chans,
                    out_channels=out_chans,
                    rbr_conv_kernel_list=[1],
                    stride=1,
                    groups=1,
                    inference_mode=inference_mode,
                    act_layer=act_layer,
                ),
            )
        else:
            self.patch_embed = nn.Identity()

        # FastViTBlock, CPE is only used on the first block
        self.blocks = nn.Sequential(
            *[
                FastViTBlock(
                    chans=out_chans,
                    num_heads=num_heads,
                    inference_mode=inference_mode,
                    act_layer=act_layer,
                    use_attn=use_attn,
                    expand_ratio=expand_ratio,
                    use_CPE=(i == 0),
                )
                for i in range(num_blocks_per_stage)
            ]
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks(x)
        return x


class FastVit(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        inference_mode: bool = False,
        in_chans_list: Tuple[int] = (48, 48, 96, 192),
        out_chans_list: Tuple[int] = (48, 96, 192, 384),
        blocks_per_stage: Tuple[int] = (2, 2, 4, 2),
        expand_ratio: Tuple[int] = (4, 4, 4, 4),
        use_attn: Tuple[bool] = (False, False, False, False),
        use_patchEmb: Tuple[bool] = (False, True, True, True),
        act_layer: nn.Module = nn.ReLU,
    ) -> None:
        """
        Constructs a FastVit model

        :param num_classes: Number of classes for classification head.
        :param inference_mode: Whether to use inference mode.
        :param in_chans_list: List of input channels for each stage.
        :param out_chans_list: List of output channels for each stage.
        :param blocks_per_stage: List of number of blocks for each stage.
        :param expand_ratio: List of expansion ratios for each stage.
        :param use_attn: List of whether to use attention for each stage.
        :param use_patchEmb: List of whether to use patch embedding for each stage.
        :param act_layer: Activation layer.
        """

        super().__init__()

        self.stem = nn.Sequential(
            RepBlock(
                in_channels=3,
                out_channels=in_chans_list[0],
                rbr_conv_kernel_list=[3, 1],
                stride=2,
                groups=1,
                inference_mode=inference_mode,
                act_layer=act_layer,
            ),
            RepBlock(
                in_channels=in_chans_list[0],
                out_channels=in_chans_list[0],
                rbr_conv_kernel_list=[3, 1],
                stride=2,
                groups=in_chans_list[0],
                inference_mode=inference_mode,
                act_layer=act_layer,
            ),
            RepBlock(
                in_channels=in_chans_list[0],
                out_channels=in_chans_list[0],
                rbr_conv_kernel_list=[1],
                stride=1,
                groups=1,
                inference_mode=inference_mode,
                act_layer=act_layer,
                skip_include_bn=True,
            ),
        )

        self.stages = nn.Sequential(
            *(
                FastStage(
                    in_chans=in_chans_list[i],
                    out_chans=out_chans_list[i],
                    num_blocks_per_stage=blocks_per_stage[i],
                    inference_mode=inference_mode,
                    use_attn=use_attn[i],
                    num_heads=8,
                    expand_ratio=expand_ratio[i],
                    act_layer=nn.ReLU,
                    use_patch_embed=use_patchEmb[i],
                )
                for i in range(len(blocks_per_stage))
            )
        )

        self.last_block = RepBlock(
            in_channels=out_chans_list[-1],
            out_channels=out_chans_list[-1],
            stride=1,
            groups=out_chans_list[-1],
            inference_mode=inference_mode,
            rbr_conv_kernel_list=[3, 1],
            act_layer=act_layer,
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=out_chans_list[-1], out_features=num_classes),
        )

    def forward(self, x) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.last_block(x)
        x = self.head(x)
        return x


def fast_vit(
    num_classes: int = 1000, inference_mode: bool = False, variant: str = "T8"
) -> nn.Module:
    """
    Constructs a FastVit model

    :param num_classes: Number of classes for classification head.
    :param inference_mode: Whether to use inference mode.
    :param variant: Variant of FastVit.
    """
    PARAMS = {
        "T8": {
            "in_chans_list": (48, 48, 96, 192),
            "out_chans_list": (48, 96, 192, 384),
            "blocks_per_stage": (2, 2, 4, 2),
            "expand_ratio": (3, 3, 3, 3),
            "use_attn": (False, False, False, False),
            "use_patchEmb": (False, True, True, True),
        },
        "T12": {
            "in_chans_list": (64, 64, 128, 256),
            "out_chans_list": (64, 128, 256, 512),
            "blocks_per_stage": (2, 2, 6, 2),
            "expand_ratio": (3, 3, 3, 3),
            "use_attn": (False, False, False, False),
            "use_patchEmb": (False, True, True, True),
        },
        "S12": {
            "in_chans_list": (64, 64, 128, 256),
            "out_chans_list": (64, 128, 256, 512),
            "blocks_per_stage": (2, 2, 6, 2),
            "expand_ratio": (4, 4, 4, 4),
            "use_attn": (False, False, False, True),
            "use_patchEmb": (False, True, True, True),
        },
        "SA12": {
            "in_chans_list": (64, 64, 128, 256),
            "out_chans_list": (64, 128, 256, 512),
            "blocks_per_stage": (2, 2, 6, 2),
            "expand_ratio": (4, 4, 4, 4),
            "use_attn": (False, False, False, True),
            "use_patchEmb": (False, True, True, True),
        },
        "SA24": {
            "in_chans_list": (64, 64, 128, 256),
            "out_chans_list": (64, 128, 256, 512),
            "blocks_per_stage": (4, 4, 12, 4),
            "expand_ratio": (4, 4, 4, 4),
            "use_attn": (False, False, False, True),
            "use_patchEmb": (False, True, True, True),
        },
        "SA36": {
            "in_chans_list": (64, 64, 128, 256),
            "out_chans_list": (64, 128, 256, 512),
            "blocks_per_stage": (6, 6, 18, 6),
            "expand_ratio": (4, 4, 4, 4),
            "use_attn": (False, False, False, True),
            "use_patchEmb": (False, True, True, True),
        },
        "MA36": {
            "in_chans_list": (76, 76, 152, 304),
            "out_chans_list": (76, 152, 304, 608),
            "blocks_per_stage": (6, 6, 18, 6),
            "expand_ratio": (4, 4, 4, 4),
            "use_attn": (False, False, False, True),
            "use_patchEmb": (False, True, True, True),
        },
    }
    if variant not in PARAMS:
        raise ValueError(
            "Invalid variant: {},valiable keys are {}".format(variant, PARAMS.keys())
        )
    variant_params = PARAMS[variant]
    return FastVit(
        num_classes=num_classes, inference_mode=inference_mode, **variant_params
    )


def reparameterize_model(model: torch.nn.Module) -> nn.Module:
    """
        Method returns a model where a multi-branched structure
        used in training is re-parameterized into a single branch
        for inference.

    :param model: MobileOne model in train mode.
    :return: MobileOne model in inference mode.
    """
    # Avoid editing original graph
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, "reparameterize"):
            module.reparameterize()
    return model

class BSConvU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
        super().__init__()
        self.with_ln = with_ln
        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise
        self.pw = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        # depthwise
        self.dw = torch.nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea
    
class DepthWiseConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_norm=False, bn_kwargs=None):
        super(DepthWiseConv, self).__init__()

        self.dw = torch.nn.Conv2d(
                in_channels=in_ch,
                out_channels=in_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=in_ch,
                bias=bias,
                padding_mode=padding_mode,
        )

        self.pw = torch.nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

    def forward(self, input):
        out = self.dw(input)
        out = self.pw(out)
        return out


class AdaptiveFM(nn.Module):
    def __init__(self, n_channels, kernel_size=3):
        super(AdaptiveFM, self).__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding=kernel_size // 2,
                              groups=n_channels)

    def forward(self, x):
        return self.conv(x) + x


class AdaptiveResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(AdaptiveResBlock, self).__init__()
        self.conv = nn.Sequential(
            BSConvU(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GELU(),
            BSConvU(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            AdaptiveFM(out_channels, kernel_size),
        )

    def forward(self, x):
        return x + self.conv(x)


class AdaptiveResGroup(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_blocks):
        super(AdaptiveResGroup, self).__init__()
        module_group = [AdaptiveResBlock(in_channels, out_channels, kernel_size=kernel_size) for _ in range(n_blocks)]
        module_group.append(BSConvU(in_channels, out_channels, kernel_size, padding=kernel_size // 2))
        module_group.append(nn.GELU())
        module_group.append(BSConvU(in_channels, out_channels, kernel_size, padding=kernel_size // 2))
        module_group.append(AdaptiveFM(out_channels, kernel_size))
        self.conv = nn.Sequential(*module_group)

    def forward(self, x):
        return x + self.conv(x)


# class SuperImageNet(nn.Module):
#     def __init__(self, inchannels=3, outchannels=3):
#         super(SuperImageNet, self).__init__()
#         hidden_dim = 64
#         scale = 2
#         # self.up = nn.Upsample(size=(384, 512), mode='bilinear')
#         self.convIn = nn.Sequential(
#             nn.Conv2d(inchannels, hidden_dim, kernel_size=3, padding=1, bias=False),
#             nn.GELU(),
#         )
#         self.convRes = AdaptiveResGroup(hidden_dim, hidden_dim, 3, 7)
#         self.convAfter = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False)
#         self.up = nn.Sequential(
#             nn.Conv2d(hidden_dim, (scale ** 2) * outchannels, kernel_size=3, padding=1, bias=False),
#             nn.PixelShuffle(scale),
#         )
#
#     def forward(self, x):
#         x = F.interpolate(x, size=(x.shape[2]//2, x.shape[3]//2), mode='bilinear')
#         x = self.convIn(x)
#         x1 = self.convAfter(self.convRes(x))
#         xOut = self.up(x + x1)
#         return xOut

class MeshRegressionNetwork(nn.Module):
    def __init__(
        self,
        inchannels: int = 3,
        inference_mode: bool = False,
        in_chans_list: Tuple[int] = (48, 48, 96, 192),
        out_chans_list: Tuple[int] = (48, 96, 192, 384),
        blocks_per_stage: Tuple[int] = (2, 2, 4, 2),
        expand_ratio: Tuple[int] = (4, 4, 4, 4),
        use_attn: Tuple[bool] = (False, False, False, False),
        use_patchEmb: Tuple[bool] = (False, True, True, True),
        act_layer: nn.Module = nn.ReLU,
    ) -> None:
        """
        Constructs a FastVit model

        :param num_classes: Number of classes for classification head.
        :param inference_mode: Whether to use inference mode.
        :param in_chans_list: List of input channels for each stage.
        :param out_chans_list: List of output channels for each stage.
        :param blocks_per_stage: List of number of blocks for each stage.
        :param expand_ratio: List of expansion ratios for each stage.
        :param use_attn: List of whether to use attention for each stage.
        :param use_patchEmb: List of whether to use patch embedding for each stage.
        :param act_layer: Activation layer.
        """

        super().__init__()
        self.patch_height = (grid_h + 1)
        self.patch_width = (grid_w + 1)

        self.stem = nn.Sequential(
            RepBlock(
                in_channels=inchannels,
                out_channels=in_chans_list[0],
                rbr_conv_kernel_list=[3, 1],
                stride=2,
                groups=1,
                inference_mode=inference_mode,
                act_layer=act_layer,
            ),
            RepBlock(
                in_channels=in_chans_list[0],
                out_channels=in_chans_list[0],
                rbr_conv_kernel_list=[3, 1],
                stride=2,
                groups=in_chans_list[0],
                inference_mode=inference_mode,
                act_layer=act_layer,
            )
            ,
            RepBlock(
                in_channels=in_chans_list[0],
                out_channels=in_chans_list[0],
                rbr_conv_kernel_list=[1],
                stride=1,
                groups=1,
                inference_mode=inference_mode,
                act_layer=act_layer,
                skip_include_bn=True,
            ),
        )

        self.stages = nn.Sequential(
            *(
                FastStage(
                    in_chans=in_chans_list[i],
                    out_chans=out_chans_list[i],
                    num_blocks_per_stage=blocks_per_stage[i],
                    inference_mode=inference_mode,
                    use_attn=use_attn[i],
                    num_heads=8,
                    expand_ratio=expand_ratio[i],
                    act_layer=act_layer,
                    use_patch_embed=use_patchEmb[i],
                )
                for i in range(len(blocks_per_stage))
            )
        )

        self.last_block = RepBlock(
            in_channels=out_chans_list[-1],
            out_channels=out_chans_list[-1],
            stride=1,
            groups=out_chans_list[-1],
            inference_mode=inference_mode,
            rbr_conv_kernel_list=[3, 1],
            act_layer=act_layer,
        )

        self.head = nn.Sequential(
            nn.Conv2d(128,1024,(3,4),2),
            nn.Flatten(),
            nn.SiLU(inplace=True),
            nn.Linear(1024, self.patch_height * self.patch_width * 2)
        )

    def forward(self, x) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.last_block(x)
        # print("x:",x.shape)
        x = self.head(x)
        return x.view(-1,self.patch_height, self.patch_width, 2)

class DownBlock(nn.Module):
    def __init__(self, inchannels, outchannels, dilation, pool=True):
        super(DownBlock, self).__init__()
        blk = []
        if pool:
            blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
        blk.append(nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1, dilation = dilation))
        blk.append(nn.ReLU(inplace=True))
        blk.append(nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1, dilation = dilation))
        blk.append(nn.ReLU(inplace=True))
        self.layer = nn.Sequential(*blk)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.layer(x)

class UpBlock(nn.Module):
    def __init__(self, inchannels, outchannels, dilation):
        super(UpBlock, self).__init__()
        #self.convt = nn.ConvTranspose2d(inchannels, outchannels, kernel_size=2, stride=2)
        self.halfChanelConv = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(outchannels*2, outchannels, kernel_size=3, padding=1, dilation = dilation),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1, dilation = dilation),
            nn.ReLU(inplace=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size = (x2.size()[2], x2.size()[3]), mode='nearest')
        x1 = self.halfChanelConv(x1)
        # print("x1:",x1.shape)
        # print("x2:",x2.shape)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

# predict the composition mask of img1
class RecNetwork(nn.Module):
    def __init__(self, inchannesl=6, outchannels=3):
        super(RecNetwork, self).__init__()

        self.down1 = DownBlock(inchannesl, 32, 1, pool=False)
        self.down2 = DownBlock(32, 64, 1)
        self.down3 = DownBlock(64, 96, 1)
        self.down4 = DownBlock(96, 128, 1)
        self.down5 = DownBlock(128, 196, 2)
        self.up1 = UpBlock(196, 128, 1)
        self.up2 = UpBlock(128, 96, 1)
        self.up3 = UpBlock(96, 64, 1)
        self.up4 = UpBlock(64, 32, 1)

        self.out = nn.Sequential(
            nn.Conv2d(32, outchannels, kernel_size=1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    # def forward(self, x1_in, x2_in):
    #     x = torch.cat([x1_in, x2_in],dim=1)
    def forward(self, x):

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        res = self.up1(x5, x4)
        res = self.up2(res, x3)
        res = self.up3(res, x2)
        res = self.up4(res, x1)
        res = self.out(res)

        return res

class RectanglingNetwork(nn.Module):
    def __init__(self, inference_mode = False):
        super(RectanglingNetwork, self).__init__()
        self.inference_mode = inference_mode
        self.ShareFeature = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16, eps=0.001, momentum=0.03),
            nn.SiLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, eps=0.001, momentum=0.03),
            nn.SiLU(inplace=True),

            nn.Conv2d(32, 3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(3, eps=0.001, momentum=0.03),
            nn.SiLU(inplace=True),
        )

        params = {
            "in_chans_list": (8, 8, 16, 32, 64, 96),
            "out_chans_list": (8, 16, 32, 64, 96, 128),
            # "in_chans_list": (8, 8, 16, 32, 64, 128),
            # "out_chans_list": (8, 16, 32, 64, 128, 196),
            "blocks_per_stage": (2, 2, 2, 2, 2, 2),
            "expand_ratio": (2, 2, 2, 2, 2, 2),
            "use_attn": (False, False, False, False, False, True),
            "use_patchEmb": (False, True, True, True, True, True),
        }
        self.meshRegression =  MeshRegressionNetwork(inchannels=3, inference_mode= False,
                                                     act_layer=nn.SiLU, **params)
        # self.superImageNet = SuperImageNet(3, 3)
        if not self.inference_mode:
            self.RecNet = RecNetwork(3,3)

    

    def forward(self, input_img, mask_img):
        batch_size, _, height, width = input_img.shape

        f_input_img = self.ShareFeature(input_img)
        feature = torch.mul(f_input_img, mask_img)

        # feature = torch.mul(input_img, mask_img)

        mesh_motion = self.meshRegression(feature)
    
        rigid_mesh = get_rigid_mesh(batch_size, height, width)
        H_one = torch.eye(3)
        H = torch.tile(H_one.unsqueeze(0), [batch_size, 1, 1]).to(device)#.cuda(non_blocking=True)
    
        ini_mesh = H2Mesh(H, rigid_mesh)
        mesh_final = ini_mesh + mesh_motion
    
        norm_rigid_mesh = get_norm_mesh(rigid_mesh, height, width)
        norm_mesh = get_norm_mesh(mesh_final, height, width)
    
        output_tps = torch_tps_transform.transformer(torch.cat((input_img, mask_img), 1), norm_rigid_mesh,  norm_mesh, (height, width))
        warp_image_final = output_tps[:, 0:3, ...]
        warp_mask_final = output_tps[:, 3:6, ...]

        warp_image_final = torch.mul(warp_image_final, warp_mask_final)
        
        # input_img_mas = torch.mul(input_img, mask_img)
        if not self.inference_mode:
            super_image = self.RecNet(warp_image_final)
        else:
            super_image = warp_image_final
        return mesh_final, warp_image_final, warp_mask_final, super_image
    


