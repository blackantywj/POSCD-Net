import random

from .transformer_utils import BertAttention, trans_nd, layer_norm
from transformers import AutoConfig
# from transformers import BertEncoder
from transformers.models.bert.modeling_bert import BertEncoder, BertLayer
from .transformer_token import TransformerMapper, Merge_attention, TransformerMapper2
from .bidirectional_cross_attention import BasicTransformerBlock
import torch
from abc import abstractmethod
# from .bert import BertLayer
import math
from typing import Tuple
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import pdb
from .fp16_util import convert_module_to_f16, convert_module_to_f32
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from .nn import (
    SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    timestep_embedding,
    checkpoint,
)
from vit.vit_transformer import ViTResNet, BasicBlock
from vit.tokenizerTrans import VT,ViT
from torchvision import models

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class BertEncodercopy(nn.Module):
    def __init__(self, config):
        super().__init__()
        layer = BertLayer(config)
        layer = nn.ModuleList([layer, condCrossAtt()])
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, input_, cond, attention_mask = None,
                output_all_encoded_layers=True):
        all_encoder_layers = []
        hidden_states = input_
        for layer_module in self.layer:
            for submodule in layer_module:             
                if type(submodule) == BertLayer:
                    hidden_states = submodule(hidden_states, attention_mask)
                else:
                    hidden_states_ct = submodule(hidden_states[0][:, 5:35], cond)
                    hidden_states = th.cat([hidden_states[0][:, 0:5], hidden_states_ct], dim = 1)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x



class TransSimpleBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        config=None,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        attention_head_size = 64
        assert self.out_channels % attention_head_size == 0
        self.in_layers = nn.Sequential(
            layer_norm(channels),
            SiLU(),
            trans_nd(config, channels, self.out_channels // attention_head_size, attention_head_size),
        )
        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            layer_norm(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                trans_nd(config, self.out_channels, self.out_channels // attention_head_size, attention_head_size),
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:

            self.skip_connection = trans_nd(config, channels, self.out_channels // attention_head_size,
                                            attention_head_size)
        else:
            self.skip_connection = nn.Sequential(nn.Linear(self.channels, self.out_channels),
                                                 nn.LayerNorm(self.out_channels, eps=config.layer_norm_eps),
                                                 )

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        # print('-'*30)
        # print(self.in_layers)
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        # print(self.in_layers, h.shape, x.shape, )
        # print(emb.shape, self.emb_layers, emb_out.shape)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out.unsqueeze(1)
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=-1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class TransModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=1,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        config=None,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        if config is None:
            config = AutoConfig.from_pretrained('bert-base-uncased')
            config.position_embedding_type = 'relative_key'
            config.max_position_embeddings = 256

            # print(self.position_embedding_type, config.max_position_embeddings)


        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        attention_head_size = 64
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    trans_nd(config, in_channels, model_channels // attention_head_size, attention_head_size)
                )
            ]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    TransformerBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        config=config,
                    )
                ]
                ch = mult * model_channels
                # if ds in attention_resolutions:
                #     layers.append(
                #         AttentionBlock(
                #             ch, use_checkpoint=use_checkpoint, num_heads=num_heads
                #         )
                #     )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims))
                )
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            TransformerBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                config=config,
            ),
            # AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads),
            TransformerBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                config=config,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    TransformerBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        config=config,
                    )
                ]
                ch = model_channels * mult
                # if ds in attention_resolutions:
                #     layers.append(
                #         AttentionBlock(
                #             ch,
                #             use_checkpoint=use_checkpoint,
                #             num_heads=num_heads_upsample,
                #         )
                #     )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        attention_head_size_final = 8
        self.out = nn.Sequential(
            layer_norm(ch),
            SiLU(),
            trans_nd(config, model_channels, out_channels // attention_head_size_final, attention_head_size_final),
        # zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        print(self.out, out_channels)

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=-1)
            h = module(cat_in, emb)
        h = h.type(x.dtype)
        return self.out(h)

    def get_feature_vectors(self, x, timesteps, y=None):
        """
        Apply the model and return all of the intermediate tensors.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: a dict with the following keys:
                 - 'down': a list of hidden state tensors from downsampling.
                 - 'middle': the tensor of the output of the lowest-resolution
                             block in the model.
                 - 'up': a list of hidden state tensors from upsampling.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result["down"].append(h.type(x.dtype))
        h = self.middle_block(h, emb)
        result["middle"] = h.type(x.dtype)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=-1)
            h = module(cat_in, emb)
            result["up"].append(h.type(x.dtype))
        return result


# class TransformerNetModel(nn.Module):
#     """
#     The full UNet model with attention and timestep embedding.

#     :param in_channels: channels in the input Tensor.
#     :param model_channels: base channel count for the model.
#     :param out_channels: channels in the output Tensor.
#     :param num_res_blocks: number of residual blocks per downsample.
#     :param attention_resolutions: a collection of downsample rates at which
#         attention will take place. May be a set, list, or tuple.
#         For example, if this contains 4, then at 4x downsampling, attention
#         will be used.
#     :param dropout: the dropout probability.
#     :param channel_mult: channel multiplier for each level of the UNet.
#     :param conv_resample: if True, use learned convolutions for upsampling and
#         downsampling.
#     :param dims: determines if the signal is 1D, 2D, or 3D.
#     :param num_classes: if specified (as an int), then this model will be
#         class-conditional with `num_classes` classes.
#     :param use_checkpoint: use gradient checkpointing to reduce memory usage.
#     :param num_heads: the number of attention heads in each attention layer.
#     """

#     def __init__(
#         self,
#         in_channels,
#         model_channels,
#         out_channels,
#         num_res_blocks,
#         attention_resolutions,
#         dropout=0,
#         channel_mult=(1, 2, 4, 8),
#         conv_resample=True,
#         dims=2,
#         num_classes=None,
#         use_checkpoint=False,
#         num_heads=1,
#         num_heads_upsample=-1,
#         use_scale_shift_norm=False,
#         config=None,
#     ):
#         super().__init__()

#         if num_heads_upsample == -1:
#             num_heads_upsample = num_heads

#         if config is None:
#             config = AutoConfig.from_pretrained('bert-base-uncased')

#         self.in_channels = in_channels
#         self.model_channels = model_channels
#         self.out_channels = out_channels
#         self.num_res_blocks = num_res_blocks
#         self.attention_resolutions = attention_resolutions
#         self.dropout = dropout
#         self.channel_mult = channel_mult
#         self.conv_resample = conv_resample
#         self.num_classes = num_classes
#         self.use_checkpoint = use_checkpoint
#         self.num_heads = num_heads
#         self.num_heads_upsample = num_heads_upsample

#         time_embed_dim = model_channels * 4
#         self.time_embed = nn.Sequential(
#             linear(model_channels, time_embed_dim),
#             SiLU(),
#             linear(time_embed_dim, time_embed_dim),
#         )

#         if self.num_classes is not None:
#             self.label_emb = nn.Embedding(num_classes, time_embed_dim)

#         attention_head_size = 64
#         self.input_blocks = nn.ModuleList(
#             [
#                 TimestepEmbedSequential(
#                     trans_nd(config, in_channels, model_channels // attention_head_size, attention_head_size)
#                 )
#             ]
#         )
#         input_block_chans = [model_channels]
#         ch = model_channels
#         ds = 1
#         for level, mult in enumerate(channel_mult):
#             for _ in range(num_res_blocks):
#                 layers = [
#                     TransSimpleBlock(
#                         ch,
#                         time_embed_dim,
#                         dropout,
#                         out_channels=mult * model_channels,
#                         dims=dims,
#                         use_checkpoint=use_checkpoint,
#                         use_scale_shift_norm=use_scale_shift_norm,
#                         config=config,
#                     )
#                 ]
#                 ch = mult * model_channels

#                 self.input_blocks.append(TimestepEmbedSequential(*layers))
#                 input_block_chans.append(ch)
#             # if level != len(channel_mult) - 1:
#             #     self.input_blocks.append(
#             #         TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims))
#             #     )
#             #     input_block_chans.append(ch)
#             #     ds *= 2

#         self.middle_block = TimestepEmbedSequential(
#             TransSimpleBlock(
#                 ch,
#                 time_embed_dim,
#                 dropout,
#                 dims=dims,
#                 use_checkpoint=use_checkpoint,
#                 use_scale_shift_norm=use_scale_shift_norm,
#                 config=config,
#             ),
#             TransSimpleBlock(
#                 ch,
#                 time_embed_dim,
#                 dropout,
#                 dims=dims,
#                 use_checkpoint=use_checkpoint,
#                 use_scale_shift_norm=use_scale_shift_norm,
#                 config=config,
#             ),
#         )

#         self.output_blocks = nn.ModuleList([])
#         print(input_block_chans)
#         for level, mult in list(enumerate(channel_mult))[::-1]:
#             for i in range(num_res_blocks ):
#                 layers = [
#                     TransSimpleBlock(
#                         ch + input_block_chans.pop(),
#                         time_embed_dim,
#                         dropout,
#                         out_channels=model_channels * mult,
#                         dims=dims,
#                         use_checkpoint=use_checkpoint,
#                         use_scale_shift_norm=use_scale_shift_norm,
#                         config=config,
#                     )
#                 ]
#                 ch = model_channels * mult

#                 # if level and i == num_res_blocks:
#                 #     layers.append(Upsample(ch, conv_resample, dims=dims))
#                 #     ds //= 2
#                 self.output_blocks.append(TimestepEmbedSequential(*layers))

#         attention_head_size_final = 8
#         self.out = nn.Sequential(
#             layer_norm(ch),
#             SiLU(),
#             zero_module(trans_nd(config, model_channels, out_channels // attention_head_size_final,
#                                             attention_head_size_final)),
#         )

#     def convert_to_fp16(self):
#         """
#         Convert the torso of the model to float16.
#         """
#         self.input_blocks.apply(convert_module_to_f16)
#         self.middle_block.apply(convert_module_to_f16)
#         self.output_blocks.apply(convert_module_to_f16)

#     def convert_to_fp32(self):
#         """
#         Convert the torso of the model to float32.
#         """
#         self.input_blocks.apply(convert_module_to_f32)
#         self.middle_block.apply(convert_module_to_f32)
#         self.output_blocks.apply(convert_module_to_f32)

#     @property
#     def inner_dtype(self):
#         """
#         Get the dtype used by the torso of the model.
#         """
#         return next(self.input_blocks.parameters()).dtype

#     def forward(self, x, timesteps, y=None):
#         """
#         Apply the model to an input batch.

#         :param x: an [N x C x ...] Tensor of inputs.
#         :param timesteps: a 1-D batch of timesteps.
#         :param y: an [N] Tensor of labels, if class-conditional.
#         :return: an [N x C x ...] Tensor of outputs.
#         """
#         assert (y is not None) == (
#             self.num_classes is not None
#         ), "must specify y if and only if the model is class-conditional"

#         hs = []
#         emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

#         if self.num_classes is not None:
#             assert y.shape == (x.shape[0],)
#             emb = emb + self.label_emb(y)

#         h = x.type(self.inner_dtype)
#         for module in self.input_blocks:
#             h = module(h, emb)
#             hs.append(h)
#         h = self.middle_block(h, emb)
#         for module in self.output_blocks:
#             # print(hs[-1].shape)
#             cat_in = th.cat([h, hs.pop()], dim=-1)
#             # print(cat_in.shape, h.shape, )
#             h = module(cat_in, emb)
#         h = h.type(x.dtype)
#         return self.out(h)

#     def get_feature_vectors(self, x, timesteps, y=None):
#         """
#         Apply the model and return all of the intermediate tensors.

#         :param x: an [N x C x ...] Tensor of inputs.
#         :param timesteps: a 1-D batch of timesteps.
#         :param y: an [N] Tensor of labels, if class-conditional.
#         :return: a dict with the following keys:
#                  - 'down': a list of hidden state tensors from downsampling.
#                  - 'middle': the tensor of the output of the lowest-resolution
#                              block in the model.
#                  - 'up': a list of hidden state tensors from upsampling.
#         """
#         hs = []
#         emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
#         if self.num_classes is not None:
#             assert y.shape == (x.shape[0],)
#             emb = emb + self.label_emb(y)
#         result = dict(down=[], up=[])
#         h = x.type(self.inner_dtype)
#         for module in self.input_blocks:
#             h = module(h, emb)
#             hs.append(h)
#             result["down"].append(h.type(x.dtype))
#         h = self.middle_block(h, emb)
#         result["middle"] = h.type(x.dtype)
#         for module in self.output_blocks:
#             cat_in = th.cat([h, hs.pop()], dim=-1)
#             h = module(cat_in, emb)
#             result["up"].append(h.type(x.dtype))
#         return result



class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.ReLU):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
                layers.append(nn.Dropout(0.1))
        self.model = nn.Sequential(*layers)


class MLP_middle(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.ReLU):
        super(MLP_middle, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
                layers.append(nn.Dropout(0.1))
        self.model = nn.Sequential(*layers)


class TransformerNetModel2(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        config=None,
        config_name='bert-base-uncased',
        training_mode='emb',
        vocab_size=None,
        experiment_mode='lm',
        init_pretrained=False,
        logits_mode=1,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if config is None:
            print(config_name)
            config = AutoConfig.from_pretrained(config_name)
            config.hidden_dropout_prob = dropout
            config.hidden_size = 512
            config.num_attention_heads = 8

        self.in_channels = in_channels  # 词向量的维度
        self.model_channels = model_channels    # 128
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks    # 2
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads  # 4
        self.num_heads_upsample = num_heads_upsample
        self.logits_mode = logits_mode  # 1
        # self.crit = LanguageModelCriterion()
        # self.lscrit = LabelSmoothing(smoothing=0.1)
        # self.bos_idx = 0
        # self.eos_idx = 0
        # self.pad_idx = 0
        # self.use_bn = 0
        # self.att_feat_size = 2048

        if training_mode == 'e2e':
            self.word_embedding = nn.Embedding(vocab_size, self.in_channels)
            if self.logits_mode == 2:
                # self.lm_head = nn.Linear(self.in_channels, vocab_size, bias=False)
                self.lm_head = nn.Linear(self.in_channels, vocab_size, bias=True)
            else:
                self.lm_head = nn.Linear(self.in_channels, vocab_size)
            with th.no_grad():
                self.lm_head.weight = self.word_embedding.weight
        elif training_mode == 'e2e-simple':
            self.word_embedding = nn.Embedding(vocab_size, self.in_channels)
            self.lm_head = nn.Linear(self.in_channels, vocab_size)
            with th.no_grad():
                self.lm_head.weight = self.word_embedding.weight

        if experiment_mode == 'conditional_gen':
            self.conditional_gen = True
            self.encoder_emb = nn.Embedding(vocab_size, config.hidden_size)
            self.encoder = BertEncoder(config)
            print(config, 'conditional_gen')
            config.is_decoder = True
            config.add_cross_attention = True
        elif experiment_mode == 'lm':
            self.conditional_gen = False

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, config.hidden_size),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # self.input_up_proj = trans_nd(config, in_channels, model_channels // attention_head_size, attention_head_size)
        self.input_up_proj_fusion = nn.Sequential(nn.Linear(in_channels, config.hidden_size),
                                              nn.SiLU(),nn.Dropout(config.hidden_dropout_prob), 
                                              nn.Linear(config.hidden_size, config.hidden_size))
        self.input_up_proj = nn.Sequential(nn.Linear(in_channels, config.hidden_size),
                                              nn.SiLU(),nn.Dropout(config.hidden_dropout_prob), nn.Linear(config.hidden_size, config.hidden_size))
        self.input_up_proj_pos = nn.Sequential(nn.Linear(in_channels, config.hidden_size),
                                           nn.SiLU(), nn.Dropout(config.hidden_dropout_prob),
                                           nn.Linear(config.hidden_size, config.hidden_size))
        # self.input_dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.input_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        if init_pretrained:
            from transformers.models.bert.modeling_bert import BertModel
            temp_bert = BertModel.from_pretrained(config_name, config=config)
            pos_temp_bert = BertModel.from_pretrained(config_name, config=config)
            del temp_bert.embeddings
            del temp_bert.pooler
            del pos_temp_bert.embeddings
            del pos_temp_bert.pooler
            self.input_transformers = temp_bert.encoder
            self.pos_input_transformers = pos_temp_bert.encoder
            print('initializing from pretrained bert.')
        else:
            print(config)
            self.input_transformers = BertEncodercopy(config)
            self.pos_input_transformers = BertEncoder(config)
            # self.input_transformers = MLP_middle((768,768*5,768*5,768))
            # self.input_transformers = TransformerMapper2(dim_embedding=768,prefix_length=25,clip_length=25,num_layers=6)

        # self.learnable_weight = nn.Parameter(torch.randn(12, 30, 48), requires_grad=True)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size) # Embedding( 512,768)
        self.pos_position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size) # Embedding( 512,768)
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        self.pos_token_type_embeddings = nn.Embedding(2, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.merge_transformer = nn.Transformer(d_model=768,nhead=16, num_encoder_layers=5, num_decoder_layers = 0, dim_feedforward=1024, activation='relu', dropout=0.1)
        self.img_project_cap = MLP((512,512*2,512*5))
        # self.img_project_pos = MLP((512,512*2,512*5))
        # self.img_project = TransformerMapper(dim_clip=768,dim_embedding=768,prefix_length=5,clip_length=5,num_layers=8)

        self.img_length = 5
        self.img_position_embeddings = nn.Embedding(self.img_length, config.hidden_size)
        self.pos_img_position_embeddings = nn.Embedding(self.img_length, config.hidden_size)
        self.img_pos_id = th.arange(self.img_length).cuda()

        self.LayerNorm_img = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pos_LayerNorm_img = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.img_dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.img_project = VT()
        # self.img_project2 = ViT(8,512,768)

        self.output_down_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                              nn.SiLU(),nn.Dropout(config.hidden_dropout_prob), nn.Linear(config.hidden_size, out_channels))
        self.output_down_proj_pos = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                              nn.SiLU(), nn.Dropout(config.hidden_dropout_prob),
                                              nn.Linear(config.hidden_size, out_channels))
        
        self.posEmbedding = nn.Embedding(14, self.in_channels)
        self.lm_pos_head = nn.Linear(self.in_channels, 14)
        with th.no_grad():
            self.lm_pos_head.weight = self.posEmbedding.weight
    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids)

    def get_pos_embeds(self, pos):
        return self.posEmbedding(pos.long())

    def get_logits(self, hidden_repr):
        if self.logits_mode == 1:
            return self.lm_head(hidden_repr)
        elif self.logits_mode == 2:
            text_emb = hidden_repr
            emb_norm = (self.lm_head.weight ** 2).sum(-1).view(-1, 1)  # vocab
            text_emb_t = th.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * th.mm(self.lm_head.weight,
                                                                     text_emb_t)  # (vocab, d) x (d, bsz*seqlen)
            scores = th.sqrt(th.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), hidden_repr.size(0),
                                                               hidden_repr.size(1)) # vocab, bsz*seqlen
            scores = -scores.permute(1, 2, 0).contiguous()

            #
            # scores1 = th.cdist(self.lm_head.weight.unsqueeze(0), hidden_repr, p=2)
            # scores1 = -scores1.permute(0, 2, 1).contiguous()
            #
            # print(scores1.shape, scores.shape)
            # print(scores1[0,0], scores[0,0])
            # print(torch.isclose(scores1, scores))

            return scores
        else:
            raise NotImplementedError

    def get_pos_logits(self, hidden_repr):
        if self.logits_mode == 1:
            return self.lm_pos_head(hidden_repr)
        elif self.logits_mode == 2:
            text_emb = hidden_repr
            emb_norm = (self.lm_pos_head.weight ** 2).sum(-1).view(-1, 1)  # vocab
            text_emb_t = th.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * th.mm(self.lm_pos_head.weight,
                                                                     text_emb_t)  # (vocab, d) x (d, bsz*seqlen)
            scores = th.sqrt(th.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), hidden_repr.size(0),
                                                               hidden_repr.size(1)) # vocab, bsz*seqlen
            scores = -scores.permute(1, 2, 0).contiguous()

            #
            # scores1 = th.cdist(self.lm_head.weight.unsqueeze(0), hidden_repr, p=2)
            # scores1 = -scores1.permute(0, 2, 1).contiguous()
            #
            # print(scores1.shape, scores.shape)
            # print(scores1[0,0], scores[0,0])
            # print(torch.isclose(scores1, scores))

            return scores
        else:
            raise NotImplementedError


    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):

        att_feats = self.pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)
        return att_feats, att_masks
    
    def pack_wrapper(self, module, att_feats, att_masks):
        if att_masks is not None:
            packed, inv_ix = self.sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
            return self.pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
        else:
            return module(att_feats)
            
    def pad_unsort_packed_sequence(self, input, inv_ix):
        tmp, _ = pad_packed_sequence(input, batch_first=True)
        tmp = tmp[inv_ix]
        return tmp
    
    def sort_pack_padded_sequence(self, input, lengths):
        sorted_lengths, indices = torch.sort(lengths, descending=True)
        tmp = pack_padded_sequence(input[indices], sorted_lengths.cpu(), batch_first=True)
        inv_ix = indices.clone()
        inv_ix[indices] = torch.arange(0,len(indices)).type_as(inv_ix)
        return tmp, inv_ix

    def forward(self, x, timesteps, img=None, y=None, poslist = None, att_feats = None, att_masks = None, pos_t = None,src_ids=None, src_mask=None, pos_cond = None, pos_start_cond = None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # print(f'real model inputs: {timesteps}')
        # att_feats, att_masks = self._prepare_feature_forward(att_feats, att_masks)
        
        # image = self.encoder(att_feats, att_masks)
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        condition_t = False

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        # emb_pos_cond = self.input_up_proj_pos_cond(pos_start)
        emb_pos = self.input_up_proj_pos(pos_t)
        emb_x = self.input_up_proj(x)
        img_inputs = self.img_project_cap(img.float()).view(-1, 5, 512)
        seq_length = x.size(1)
        pos_length = pos_t.size(1)
        position_ids = self.position_ids[:, 5 : seq_length + 5]
        pos_position_ids = self.position_ids[:, 5 : pos_length + 5]
        # print(emb_x.shape, emb.shape, self.position_embeddings)
        # emb_x.shape=batch*seq_len*dim;position_ids.shape=1*24;pos_emb(pos_id).shape=1*24*768;emb.shape=128*768
        pos_token_shape = emb_pos.shape[:-1]
        text_token_shape = emb_x.shape[:-1]
        pos_token_type_ids = torch.ones(pos_token_shape, dtype=torch.long, device="cuda")
        text_token_type_ids = torch.ones(text_token_shape, dtype=torch.long, device="cuda")
        text_token_type_embedding = self.token_type_embeddings(text_token_type_ids)
        pos_token_type_embedding = self.pos_token_type_embeddings(pos_token_type_ids)
        emb_inputs = self.position_embeddings(position_ids) + emb_x + emb.unsqueeze(1).expand(-1, seq_length, -1) + text_token_type_embedding
        emb_pos_inputs = self.pos_position_embeddings(pos_position_ids) + emb_pos + emb.unsqueeze(1).expand(-1, pos_length, -1) + pos_token_type_embedding
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))
        emb_pos_inputs = self.dropout(self.LayerNorm(emb_pos_inputs))

        visiual_token_shape = img_inputs.shape[:-1]
        visual_token_type_ids = torch.zeros(visiual_token_shape, dtype=torch.long, device="cuda")
        visual_token_type_embedings = self.token_type_embeddings(visual_token_type_ids)
        pos_visual_token_type_embedings = self.pos_token_type_embeddings(visual_token_type_ids)
        img_inputs = img_inputs + visual_token_type_embedings + self.img_position_embeddings(self.img_pos_id)
        pos_img_inputs = img_inputs + pos_visual_token_type_embedings + self.pos_img_position_embeddings(self.img_pos_id)
        img_inputs = self.img_dropout(self.LayerNorm_img(img_inputs))
        pos_img_inputs = self.img_dropout(self.pos_LayerNorm_img(pos_img_inputs))
        # emb_pos_inputs = th.zeros(emb_pos_inputs.shape, device=emb_pos_inputs.device)
        emb_inputs = th.cat([img_inputs, emb_inputs], dim=1)
        pos_emb_inputs = th.cat([pos_img_inputs, emb_pos_inputs], dim=1)
        # emb_inputs = self.merge_attention(emb_inputs,emb_inputs,emb_inputs)
        if condition_t:
            encoder_attention_mask = torch.ones(pos_cond.shape[:-1])
            # print(emb_inputs.shape, encoder_hidden_states.shape, encoder_attention_mask.shape)
            input_trans_hidden_states = self.input_transformers(emb_inputs,
                                                                encoder_hidden_states=pos_cond,
                                                                encoder_attention_mask=encoder_attention_mask,
                                                                ).last_hidden_state
            pos_input_trans_hidden_states = self.pos_input_transformers(pos_emb_inputs).last_hidden_state
        else:
            pos_start_cond = self.input_up_proj_fusion(pos_start_cond)       
            input_trans_hidden_states = self.input_transformers(emb_inputs, pos_start_cond)
            pos_input_trans_hidden_states = self.pos_input_transformers(pos_emb_inputs).last_hidden_state
            # input_trans_hidden_states = self.input_transformers(emb_inputs)
        h_pos = self.output_down_proj_pos(pos_input_trans_hidden_states[: , 5:35])   #[:,1:]
        h = self.output_down_proj(input_trans_hidden_states[9][: ,5:35])
        h = h.type(x.dtype)
        h_pos = h_pos.type(pos_t.dtype)
        return h_pos, h #,img
    
    def get_feature_vectors(self, x, timesteps, y=None):
        """
        Apply the model and return all of the intermediate tensors.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: a dict with the following keys:
                 - 'down': a list of hidden state tensors from downsampling.
                 - 'middle': the tensor of the output of the lowest-resolution
                             block in the model.
                 - 'up': a list of hidden state tensors from upsampling.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result["down"].append(h.type(x.dtype))
        h = self.middle_block(h, emb)
        result["middle"] = h.type(x.dtype)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=-1)
            h = module(cat_in, emb)
            result["up"].append(h.type(x.dtype))
        return result

import copy
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
    kld = -0.5 * torch.sum(1 + (recog_logvar - prior_logvar)
                           - torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar))
                           - torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar)), dim=-1)
    return kld

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Latent(nn.Module):
    def __init__(self,dim):
        super(Latent, self).__init__()
        self.mean = PositionwiseFeedForward(dim, 2048, dropout=0)
        self.mean_1 = PositionwiseFeedForward(dim, 2048, dropout=0)
        self.var = PositionwiseFeedForward(dim, 2048, dropout=0)
        self.var_1 = PositionwiseFeedForward(dim, 2048, dropout=0)
        # self.mean_p = PositionwiseFeedForward_11(dim*2, 2048, dropout=0.1)
        # self.var_p = PositionwiseFeedForward_11(dim*2, 2048, dropout=0.1)
    def forward(self, x, x_p, train=True):
        mean = self.mean(x)
        log_var = self.var(x)
        eps = torch.randn(x.size())
        std = torch.exp(0.5 * log_var)
        eps = eps.cuda()
        z = eps * std + mean
        kld_loss = 0
        if x_p is not None:
            x_p = x_p.squeeze(1)
            mean_p = self.mean_1(x_p)
            log_var_p = self.var_1(x_p)
            kld_loss = gaussian_kld(mean_p, log_var_p, mean, log_var)
            kld_loss = torch.mean(kld_loss)
        if train:
            std = torch.exp(0.5 * log_var_p)
            eps = eps.cuda()
            z = eps * std + mean_p
            # z =  mean_p
        return kld_loss, z
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # kernel, projection
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
    
class PositionwiseFeedForward_11(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward_11, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model // 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class AverageSelfAttention(nn.Module):
    def __init__(self, attention_size):
        super(AverageSelfAttention, self).__init__()
        w = torch.empty(attention_size)
        nn.init.normal_(w, std=0.02)
        self.attention_weights = nn.Parameter(w)
        self.softmax = nn.Softmax(dim=-1)
        self.non_linearity = nn.GELU()

    def forward(self, inputs, attention_mask=None):
        scores = self.non_linearity(inputs.matmul(self.attention_weights))
        if attention_mask is not None:
            scores = scores + attention_mask
        scores = self.softmax(scores)
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))
        representations = weighted.sum(1).squeeze(1)

        return representations
    
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class posEmbeddings(nn.Module):
    
    def __init__(self, d_model, vocab):
        super(posEmbeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        # torch.nn.init.normal_(self.lut.weight)
        self.d_model = d_model
        self.layernom = LayerNorm(self.d_model)
        self.dropout = nn.Dropout(p=0.1)
        pe = torch.zeros(5000, d_model)
        position = torch.arange(0, 5000).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, decodemask, x):
        # 对于POS forword Embedding时可能会有问题
        if decodemask == None:
            a = self.lut(x.clone()) * math.sqrt(self.d_model)
            return self.dropout(self.layernom(a + self.pe[:, :a.size(1)]))
        else:
            a = self.lut(decodemask.clone()) * math.sqrt(self.d_model)
            a = self.layernom(a + self.pe[:, :a.size(1)] + x)
            a = self.dropout(a)
            return a
        
class posEncoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(posEncoder, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.latent = Latent(layer.size)
        self.houyan1 = MultiHeadedAttention(8, 512, dropout=0.1)
        self.houyan2 = MultiHeadedAttention(8, 512, dropout=0.1)
        self.av1 = AverageSelfAttention(layer.size)
        self.av2 = AverageSelfAttention(layer.size)

        # self.pos_up_project = nn.Sequential(nn.Linear(24, layer.size),
        #                         nn.SiLU(), nn.Dropout(0.1), nn.Linear(layer.size, layer.size))
    ''''
    pos_h, memory (VN), src_mask, pos_mask, target_mask, train
    '''
    def forward(self, x,  memory, src_mask, seq_mask, tgt_mask, train=True):
        if train == True:
            x_norm1 = self.houyan1(x, x, x, seq_mask)
            x_norm = self.dropout(x_norm1 + x)
            x_norm1 = self.houyan2(x_norm, memory, memory, src_mask)
            x_norm = self.dropout(x_norm1 + x_norm)
            x_norm = self.av1(x_norm, seq_mask.squeeze(1))
            x_norm_p = self.av2(memory, src_mask.squeeze(1))
            kl_loss, z = self.latent(x_norm_p, x_norm, train)
        if train == False and x is not None :
            x_norm1 = self.houyan1(x, x, x, seq_mask)
            x_norm = self.dropout(x_norm1+x)
            x_norm1 = self.houyan2(x_norm,memory,memory,src_mask)
            x_norm = self.dropout(x_norm1 + x_norm)
            x_norm = self.av1(x_norm,seq_mask.squeeze(1)) # houyan
            x_norm_p = self.av2(memory,src_mask.squeeze(1)) # xianyan
            kl_loss, z = self.latent(x_norm_p, x_norm, True)
            return 0, kl_loss, z.unsqueeze(1)
        if x is None:
            x_norm_p = self.av2(memory, None)
            x_norm = None
            kl_loss, z = self.latent(x_norm_p, x_norm, train=False)
            return 0, kl_loss, z.unsqueeze(1)
        # a = torch.cat((x_p,z),dim=-1)
        # z1 = z.unsqueeze(1).repeat(1,32,1)
        # x = z1 + x
        # x = self.norm1(x)
        # for layer in self.layers:
        #     x = layer(x, memory.detach(), src_mask, target_mask)
        return 0, kl_loss, z.unsqueeze(1)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask=None):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x,tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m,src_mask))
        return self.sublayer[2](x, self.feed_forward)
    
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask, reduction='mean'):
        if target.ndim == 3:
            target = target.reshape(-1, target.shape[2])
            mask = mask.reshape(-1, mask.shape[2])
        N,L = input.shape[:2]
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)].to(input)

        output = -input.gather(2, target.unsqueeze(2).to(torch.int64)).squeeze(2) * mask

        if reduction == 'none':
            output = output.view(N,L).sum(1) / mask.view(N,L).sum(1)
        elif reduction == 'mean':
            output = torch.sum(output) / torch.sum(mask)

        return output

class condCrossAttgroup(nn.Module):
    def __init__(self, num):
        super(condCrossAttgroup, self).__init__()
        cca = condCrossAtt()
        self.layer = nn.ModuleList([copy.deepcopy(cca)
                                    for _ in range(num)])
    def forward(self, x, cond):
        for sublayer in self.layer:
            output = sublayer(x, cond)
            x = output
        return x

class condCrossAtt(nn.Module):
    def __init__(self):
        super(condCrossAtt, self).__init__()
        self.conMHA = MultiHeadedAttention(8, 512, dropout=0.1)
        self.LayerNorm1 = nn.LayerNorm(512,eps=1e-12)
        self.dropout1 = nn.Dropout(p=0.1)
        self.FFN = PositionwiseFeedForward(512, 512, dropout=0.1)
        self.LayerNorm2 = nn.LayerNorm(512,eps=1e-12)
        self.dropout2 = nn.Dropout(p=0.1)
    def forward(self, x, pos):
        _x = x
        x = self.conMHA(x, pos, pos)
        x = self.LayerNorm1(x + _x)
        x = self.dropout1(x)
        _x = x
        x = self.FFN(x)
        x = self.LayerNorm2(x + _x)
        x = self.dropout2(x)
        return x

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size=0, padding_idx=0, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        # self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # self.size = size
        self.true_dist = None

    def forward(self, input, target, mask, reduction='mean'):
        N,L = input.shape[:2]
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]

        input = input.reshape(-1, input.size(-1))
        target = target.reshape(-1)
        mask = mask.reshape(-1).to(input)

        # assert x.size(1) == self.size
        self.size = input.size(1)
        # true_dist = x.data.clone()
        true_dist = input.data.clone()
        # true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.fill_(self.smoothing / (self.size - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # true_dist[:, self.padding_idx] = 0
        # mask = torch.nonzero(target.data == self.padding_idx)
        # self.true_dist = true_dist
        output = self.criterion(input, true_dist).sum(1) * mask

        if reduction == 'none':
            output = output.view(N,L).sum(1) / mask.view(N,L).sum(1)
        elif reduction == 'mean':
            output = torch.sum(output) / torch.sum(mask)

        return output

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)