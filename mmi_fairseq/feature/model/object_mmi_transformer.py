# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: transformer_encoder
@time: 2020/11/18 11:35
@desc: Transformer encoder with src-tokens and img-features as inputs

"""

from typing import Optional

import torch
import torch.nn as nn
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    TransformerModel,
    TransformerEncoder,
    EncoderOut,
    base_architecture as transformer_base_architecture
)

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024

@register_model("mmi-obj-transformer")
class MMIObjectTransformerModel(TransformerModel):
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
        self.final = nn.Linear(in_features=args.encoder_embed_dim+args.img_dim, out_features=1, bias=True)
        #self.final = nn.Linear(in_features=args.encoder_embed_dim, out_features=args.img_dim, bias=True)
        #self.cos = nn.CosineSimilarity(dim=2)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerModel.add_args(parser)
        parser.add_argument('--img-dim', type=int, metavar='N', default=2048,
                            help='image feature dimension')

    def forward(self, src_tokens, src_label, objs, objs_mask, src_lengths, prev_output_tokens, **kwargs):
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
            src_label (LongTensor): positive example or negative example
                `(batch)`
            objs (FloatTensor): images features in the source sentences
                `(batch, max_obj, feature_dim)`
            objs_mask (FloatTensor): mask file `(batch, max_obj)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing

        Returns:
            output_: image_feature * text_feature, shape `(batch, sent_len)`
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        '''
        x = self.final(encoder_out.encoder_out).transpose(0, 1) # T * B * C -> B * T * C
        src_imgs = torch.unsqueeze(src_imgs, dim=1)
        src_imgs = src_imgs.expand(x.shape[0], x.shape[1], x.shape[2])
        #print(src_label)
        #print(self.cos(x, src_imgs))
        #output_ = torch.nn.functional.sigmoid(torch.matmul(x, src_imgs).squeeze(dim=-1)) * (1-encoder_out.encoder_padding_mask.float()) # B * T
        output_ = torch.nn.functional.sigmoid(self.cos(x, src_imgs)) * (1-encoder_out.encoder_padding_mask.float())  # B * T
        #print(output_.sum(dim=-1)/src_lengths)
        #print(output_)
        return output_.sum(dim=-1)/src_lengths, src_label
        '''
        x = encoder_out.encoder_out.transpose(0, 1) # T * B * C -> B * T * C
        object_mask = objs_mask.unsqueeze(dim=-1) # B * max_obj * feature_dim
        objs = objs * object_mask # B * max_obj * feature_dim
        src_imgs = objs.sum(dim=1)/objs_mask.sum(dim=-1).unsqueeze(dim=-1) # B * feature_dim
        src_imgs = torch.unsqueeze(src_imgs, dim=1)
        src_imgs = src_imgs.expand(x.shape[0], x.shape[1], src_imgs.shape[2])
        feature = torch.nn.functional.sigmoid(self.final(torch.cat((x, src_imgs), dim=-1)).squeeze(dim=-1)) * (1-encoder_out.encoder_padding_mask.float()) # B * T        
        return feature.sum(dim=-1)/src_lengths, src_label

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return MMIObjectTransformerEncoder(args, src_dict, embed_tokens)


class MMIObjectTransformerEncoder(TransformerEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)

        self.img_dim = args.img_dim

    def forward_embedding(
        self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def forward(
        self,
        src_tokens,
        src_lengths,
        cls_input=None,
        return_all_hiddens=False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

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
        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )


@register_model_architecture('mmi-obj-transformer', 'baseline-mmi-obj-transformer')
def base_architecture(args):
    transformer_base_architecture(args)

