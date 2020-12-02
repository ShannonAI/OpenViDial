# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: object_transformer
@time: 2020/12/1 10:20
@desc: 

"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    TransformerModel,
    TransformerEncoder,
    TransformerDecoder,
    EncoderOut,
    base_architecture as transformer_base_architecture
)

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model("obj-transformer")
class ObjTransformerModel(TransformerModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

    def forward(self, src_tokens, objs, objs_mask, src_lengths, prev_output_tokens, **kwargs):
        """
        Run the forward pass for an encoder-decoder model.

        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., teacher forcing) to
        the decoder to produce the next outputs::

            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            objs (FloatTensor): objects features in the source sentences
                `(batch, sent_num. max_obj, dim)`
            objs_mask (BoolTensor): objects mask in the source sentences
                `(batch, sent_num, max_obj)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        encoder_out = self.encoder(src_tokens, objs, objs_mask, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return decoder_out

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return ObjTransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, 'no_cross_attention', False),
        )


class ObjTransformerEncoder(TransformerEncoder):
    """
    Object Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        # sep = torch.LongTensor(dictionary.sep())
        # self.register_buffer("sep", sep)
        self.token_type_embedding = nn.Embedding(2, args.encoder_embed_dim)  # image/token
        self.image_proj = nn.Linear(2048, args.encoder_embed_dim)

    def forward_embedding(self, src_tokens, objs):
        bsz, token_length = src_tokens.size()
        _, sent_num, max_obj, dim = objs.size()

        # embed tokens
        x = embed = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        x += self.token_type_embedding(torch.ones_like(src_tokens))

        # embed objects
        # [bsz, sent_num*max_obj, dim]
        img_type_emb = self.token_type_embedding(torch.zeros_like(objs[:, :, :, 0]).long()).view(bsz, sent_num*max_obj, -1)
        img_obj_emb = self.image_proj(objs.view(bsz, sent_num*max_obj, -1))
        y = img_obj_emb + img_type_emb
        if self.embed_positions is not None:
            # [bsz, sent_num*max_obj, dim]  todo(yuxian): rethink about this position embedding
            # img_pos_emb = self.embed_positions(torch.ones_like(objs[:, :, 0, 0].long())).expand(-1, sent_num*max_obj, -1)
            img_pos_emb = self.embed_positions(torch.ones_like(objs[:, :, 0, 0].long())).repeat(1, max_obj, 1)
            y = y + img_pos_emb

        # bsz, sent_num*max_obj + token_length, dim
        x = torch.cat([y, x], dim=1)

        if self.layernorm_embedding:
            x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # return x, embed
        return x, None

    def forward(self, src_tokens, objs, objs_mask, src_lengths, cls_input=None, return_all_hiddens=False, **unused):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            objs (FloatTensor): objects features in the source sentences
                `(batch, sent_num. max_obj, dim)`
            objs_mask (BoolTensor): objects mask in the source sentences
                `(batch, sent_num, max_obj)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        if self.layer_wise_attention:
            return_all_hiddens = True

        x, encoder_embedding = self.forward_embedding(src_tokens, objs)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        bsz = src_tokens.size(0)
        encoder_padding_mask = torch.cat([objs_mask.view(bsz, -1), src_tokens.eq(self.padding_idx)], dim=-1)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if not self.training or (dropout_probability > self.encoder_layerdrop):
                x = layer(x, encoder_padding_mask)
                if return_all_hiddens:
                    encoder_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)
            if return_all_hiddens:
                encoder_states[-1] = x

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
        )

    def max_positions(self):  # todo(yuxian) 搞清楚在哪里用
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())


@register_model_architecture('obj-transformer', 'baseline-obj-transformer')
def base_architecture(args):
    transformer_base_architecture(args)

