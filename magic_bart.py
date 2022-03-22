import logging
import math
import random
from typing import Optional, Tuple, Union, Dict, Any, List
from operator import itemgetter

import dgl
from dgl.nn import SAGEConv
from transformers.file_utils import ModelOutput
from transformers.integrations import TensorBoardCallback
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase

import torch
from torch.nn import CrossEntropyLoss
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.activations import ACT2FN
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput, Seq2SeqModelOutput, \
    BaseModelOutputWithPastAndCrossAttentions
from dataclasses import dataclass
from transformers.models.bart.modeling_bart import (
    shift_tokens_right,
    BartConfig,
    BartEncoder,
    BartPretrainedModel,
    _expand_mask, _make_causal_mask,
    BartLearnedPositionalEmbedding, BartAttention,
)
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from packaging import version

from magic_hgt import HGT as MagicHGT

if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast
from run_mybart import model_args
from dgl import DGLGraph

logger = logging.getLogger(__name__)

node_type_mapping = {'search_news': 0, 'search_point': 0, 'search_xueqiu': 0, 'search_qa': 1,
                             'search_qa_answer': 2, 'query': 3, 'comment': 4, 'search_news_sent': 5, 'search_point_sent': 5,
                             'search_xueqiu_sent': 5, 'pad': 6}
ALL_NODE_TYPE = 4
if model_args.expand_sentence_node: ALL_NODE_TYPE += 1
if model_args.add_comment: ALL_NODE_TYPE += 1
edge_list = ['3|3', '3|0', '3|1', '1|2', '3|2']
if model_args.add_bidirectional_edge: edge_list.extend(['0|3', '1|3', '2|1', '2|3'])
if model_args.expand_sentence_node: edge_list.append('0|5')
if model_args.add_bidirectional_edge and model_args.expand_sentence_node: edge_list.append('5|0')
if model_args.add_comment: edge_list.append('0|4')
if model_args.add_bidirectional_edge and model_args.add_comment: edge_list.append('4|0')
edge_mapping = {e: eid for eid, e in enumerate(edge_list)}

class MyBartDecoder(BartPretrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`BartDecoderLayer`

    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            self.padding_idx,
        )
        self.layers = nn.ModuleList([MyBartDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        encoder_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        addi_source_encoder_hidden_states=None,
        addi_source_encoder_attention_mask=None,
        graph_hidden_states: Optional[torch.Tensor]=None,
        graph_hidden_states_mask: Optional[torch.Tensor]=None,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            encoder_hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`, `optional`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, encoder_sequence_length)`, `optional`):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the heas is **masked**.

            encoder_head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules in encoder to avoid performing cross-attention
                on hidden heads. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the heas is **masked**.

            past_key_values (:obj:`Tuple[Tuple[torch.Tensor]]` of length :obj:`config.n_layers` with each tuple having 2 tuples each of which has 2 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up
                decoding.

                If :obj:`past_key_values` are used, the user can optionally input only the last
                :obj:`decoder_input_ids` (those that don't have their past key value states given to this model) of
                shape :obj:`(batch_size, 1)` instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size,
                sequence_length)`.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # expand encoder attention mask
        if addi_source_encoder_hidden_states is not None and addi_source_encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            addi_source_encoder_attention_mask = _expand_mask(addi_source_encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # expand encoder attention mask
        if graph_hidden_states is not None and graph_hidden_states_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            graph_hidden_states_mask = _expand_mask(graph_hidden_states_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    encoder_head_mask[idx] if encoder_head_mask is not None else None,
                    None,
                    addi_source_encoder_hidden_states,
                    addi_source_encoder_attention_mask,
                    graph_hidden_states,
                    graph_hidden_states_mask,
                )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    encoder_layer_head_mask=(encoder_head_mask[idx] if encoder_head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    addi_source_encoder_hidden_states=addi_source_encoder_hidden_states,
                    addi_source_encoder_attention_mask=addi_source_encoder_attention_mask,
                    graph_hidden_states=graph_hidden_states,
                    graph_hidden_states_mask=graph_hidden_states_mask,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class MyBartDecoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        logger.info(f'============= addi_share_decoder_attn {model_args.addi_share_decoder_attn} =============')
        if model_args.addi_share_decoder_attn:
            self.addi_source_attn = self.encoder_attn
            self.graph_attn = self.encoder_attn
        else:
            self.addi_source_attn = BartAttention(
                self.embed_dim,
                config.decoder_attention_heads,
                dropout=config.attention_dropout,
                is_decoder=True,
            )
            self.graph_attn = BartAttention(
                self.embed_dim,
                config.decoder_attention_heads,
                dropout=config.attention_dropout,
                is_decoder=True,
            )
        logger.info(f'============= two_hidden_merge_method {model_args.two_hidden_merge_method} =============')
        if model_args.two_hidden_merge_method == 'cat':
            self.merge_two_hidden = nn.Linear(config.decoder_ffn_dim, int(config.decoder_ffn_dim / 2))
        elif model_args.two_hidden_merge_method == 'add':
            pass
        elif model_args.two_hidden_merge_method == 'weightsum':
            self.bi_weight_layer = nn.Bilinear(int(config.decoder_ffn_dim / 2), int(config.decoder_ffn_dim / 2), 1)
        elif model_args.two_hidden_merge_method == 'fusion':
            self.fusion_layer = nn.Linear(config.decoder_ffn_dim * 2, int(config.decoder_ffn_dim / 2))
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # self.addi_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        encoder_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        addi_source_encoder_hidden_states = None,
        addi_source_encoder_attention_mask = None,
        graph_hidden_states: Optional[torch.Tensor] = None,
        graph_hidden_states_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (:obj:`torch.FloatTensor`): cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (:obj:`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(config.encoder_attention_heads,)`.
            encoder_layer_head_mask (:obj:`torch.FloatTensor`): mask for encoder attention heads in a given layer of
                size `(config.encoder_attention_heads,)`.
            past_key_value (:obj:`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=encoder_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            addi_source_hidden_states, _, _ = self.addi_source_attn(
                hidden_states=residual,
                key_value_states=addi_source_encoder_hidden_states,
                attention_mask=addi_source_encoder_attention_mask,
                layer_head_mask=None,
                past_key_value=None,
                output_attentions=output_attentions,
            )
            graph_attn_hidden_states, _, _ = self.graph_attn(
                hidden_states=residual,
                key_value_states=graph_hidden_states,
                attention_mask=graph_hidden_states_mask,
                layer_head_mask=None,
                past_key_value=None,
                output_attentions=output_attentions,
            )
            addi_source_hidden_states += graph_attn_hidden_states
            # addi_source_hidden_states = self.addi_attn_layer_norm(addi_source_hidden_states)
            # hidden_states [batch, dec_length, 1024]
            if model_args.two_hidden_merge_method == 'cat':
                hidden_states = self.merge_two_hidden(torch.cat([hidden_states, addi_source_hidden_states], dim=-1))
            elif model_args.two_hidden_merge_method == 'add':
                hidden_states = hidden_states + addi_source_hidden_states
            elif model_args.two_hidden_merge_method == 'weightsum':
                score = F.sigmoid(self.bi_weight_layer(hidden_states, addi_source_hidden_states))
                # logger.info(score.mean())
                hidden_states = score * hidden_states + (1 - score) * addi_source_hidden_states
            elif model_args.two_hidden_merge_method == 'fusion':
                fusion = torch.cat([
                    hidden_states,
                    addi_source_hidden_states,
                    addi_source_hidden_states + hidden_states,
                    addi_source_hidden_states * hidden_states,
                ], dim=-1)
                hidden_states = self.fusion_layer(fusion)
            # hidden_states = addi_source_hidden_states

            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class MyBartModel(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        # self.another_encoder = BartEncoder(config, self.shared)
        self.decoder = MyBartDecoder(config, self.shared)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        addi_source=None,
        addi_source_attention_mask=None,
        addi_source_encoder_outputs=None,
        graph_hidden_states: Optional[torch.Tensor]=None,
        graph_hidden_states_mask: Optional[torch.Tensor]=None,
    ):
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            addi_source_encoder_outputs = self.encoder(
                input_ids=addi_source,
                attention_mask=addi_source_attention_mask
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
            addi_source_encoder_outputs = BaseModelOutput(
                last_hidden_state=addi_source_encoder_outputs[0],
                hidden_states=addi_source_encoder_outputs[1] if len(addi_source_encoder_outputs) > 1 else None,
                attentions=addi_source_encoder_outputs[2] if len(addi_source_encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            encoder_head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            addi_source_encoder_hidden_states=addi_source_encoder_outputs[0],
            addi_source_encoder_attention_mask=addi_source_attention_mask,
            graph_hidden_states=graph_hidden_states,
            graph_hidden_states_mask=graph_hidden_states_mask,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class MagicHeteroGraph(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.activation_function]
        node_dict = {str(i): i for i in range(ALL_NODE_TYPE)}
        if model_args.magic_hgt:
            self.layers = MagicHGT(node_dict, edge_mapping, config.d_model, config.d_model, config.d_model, model_args.graph_layer_num, config.num_attention_heads)
        # else:
        #     self.layers = HGT(node_dict, edge_mapping, config.d_model, config.d_model, config.d_model, model_args.graph_layer_num, config.num_attention_heads)
        self.score_predict = nn.Linear(in_features=config.d_model, out_features=1)
        self.score_predict_loss = nn.MSELoss()
        if model_args.graph_node_residual:
            self.merge_original = nn.Sequential(nn.Linear(in_features=2*config.d_model, out_features=config.d_model), nn.Tanh())
            # self.merge_original = Fusion(feat_size=config.d_model, out_size=config.d_model, activation=nn.Tanh())

    def forward(self, graph: dgl.DGLHeteroGraph) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        device = graph.device
        if model_args.magic_hgt:
            query = graph.nodes['3'].data['bart_hidden']
            graph: dgl.DGLHeteroGraph = self.layers(graph, query)
        else:
            graph: dgl.DGLHeteroGraph = self.layers(graph)
        graph_states = []
        graph_original_states = []
        graph_scores = []
        for g in dgl.unbatch(graph):  # type: dgl.DGLHeteroGraph
            states = []
            original_states = []
            scores = []
            for nt in graph.ntypes:
                states.append(g.nodes[nt].data['bart_hidden'])
                scores.append(g.nodes[nt].data['score'])
                if model_args.graph_node_residual:
                    original_states.append(g.nodes[nt].data['bart_hidden_original'])
            graph_states.append(torch.cat(states, dim=0))
            graph_scores.append(torch.cat(scores, dim=0))
            if model_args.graph_node_residual:
                graph_original_states.append(torch.cat(original_states, dim=0))
        max_nodes = max([s.shape[0] for s in graph_states])
        max_nodes = min([max_nodes, 50])
        graph_hidden_states = torch.ones([len(graph_states), max_nodes, graph_states[0].shape[1]]).to(device)
        graph_hidden_states_mask = torch.zeros([len(graph_states), max_nodes]).to(device)
        graph_pad_scores = torch.zeros([len(graph_states), max_nodes]).to(device)
        if model_args.graph_node_residual:
            graph_original_hidden_states = torch.ones([len(graph_original_states), max_nodes, graph_original_states[0].shape[1]]).to(device)
        for idx, s in enumerate(graph_states):
            graph_hidden_states[idx, :s.shape[0]] = s[:max_nodes]
            graph_hidden_states_mask[idx, :s.shape[0]] = 1
            graph_pad_scores[idx, :len(graph_scores[idx])] = graph_scores[idx]
            if model_args.graph_node_residual:
                graph_original_hidden_states[idx, :s.shape[0]] = graph_original_states[idx][:max_nodes]
        score_predict_loss = self.score_predict_loss(self.score_predict(graph_hidden_states).squeeze(dim=-1), graph_pad_scores) * graph_hidden_states_mask
        score_predict_loss = score_predict_loss.mean()
        if model_args.graph_node_residual:
            graph_hidden_states = self.merge_original(torch.cat([graph_hidden_states, graph_original_hidden_states], dim=-1))
            # graph_hidden_states = self.merge_original(graph_hidden_states, graph_original_hidden_states)
        return graph_hidden_states, graph_hidden_states_mask, score_predict_loss


class MyBart(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"lm_head\.weight",
    ]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = MyBartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        if model_args.hetero_graph:
            if model_args.hetero_graph_model == 'hgt':
                self.magic_graph = MagicHeteroGraph(config)
            # elif model_args.hetero_graph_model == 'han':
            #     self.magic_graph = MagicHANGraph(config)
            # else:
            #     raise NotImplementedError(f'heterogenous graph model {model_args.hetero_graph_model} not implemented')
        # else:
        #     self.magic_graph = MagicGraph(config)
        self.init_weights()

    def build_edges(self, edges: dgl.udf.EdgeBatch, device):
        edge_ids = []
        for src_type, dst_type in zip(edges.src['type'], edges.dst['type']):
            edge_ids.append(edge_mapping[f"{src_type}|{dst_type}"])
        return torch.tensor(edge_ids).to(device)

    def build_graph(self, source_node_id: List[List], target_node_id: List[List], node_txt_ids: List[List], node_txt_attention_mask: List[List], node_type: List[List], node_scores: List[List]) -> dgl.DGLGraph:
        # 构造graph
        graph_list = []
        for i in range(len(node_txt_ids)):
            g = dgl.graph((source_node_id[i], target_node_id[i]))  # type: DGLGraph
            g.ndata['txt_ids'] = torch.tensor(node_txt_ids[i])  # [:g.num_nodes()]
            g.ndata['txt_attention_mask'] = torch.tensor(node_txt_attention_mask[i])  # [:g.num_nodes()]
            g.ndata['type'] = torch.tensor(node_type[i])
            g.ndata['score'] = torch.tensor(node_scores[i])
            g.ndata['batch_idx'] = torch.tensor([i] * len(node_type[i]))
            graph_list.append(g)
        encoder = self.get_encoder()
        graph = dgl.batch(graph_list).to(encoder.device)  # type: DGLGraph
        # batch_num_nodes = graph.batch_num_nodes()
        # batch_num_edges = graph.batch_num_edges()
        graph_encoder_outputs = encoder(
            input_ids=graph.ndata['txt_ids'],
            attention_mask=graph.ndata['txt_attention_mask']
        )
        if model_args.node_interaction_w_query:
            graph.ndata['bart_hidden_w_len'] = graph_encoder_outputs.last_hidden_state
        else:
            graph.ndata['bart_hidden'] = torch.mean(graph_encoder_outputs.last_hidden_state, dim=1)

        if model_args.graph_node_residual:
            graph.ndata['bart_hidden_original'] = torch.mean(graph_encoder_outputs.last_hidden_state, dim=1)
        del graph.ndata['txt_ids']
        del graph.ndata['txt_attention_mask']
        if model_args.hetero_graph:
            graph_list = []
            for i, g in enumerate(dgl.unbatch(graph)):
                if model_args.node_interaction_w_query:
                    g.ndata['bart_hidden'] = torch.zeros([g.num_nodes(), graph_encoder_outputs.last_hidden_state.size()[-1]]).to(encoder.device)
                    query_node_repre = g.ndata['bart_hidden_w_len'][g.filter_nodes(lambda nodes: (nodes.data['type'] == 3))]  # [1, sent_len, dim] type: torch.Tensor
                    doc_node_repre = g.ndata['bart_hidden_w_len'][g.filter_nodes(lambda nodes: (nodes.data['type'] != 3))]  # [node_num, sent_len, dim] type: torch.Tensor
                    query_node_repre = torch.mean(query_node_repre, dim=1, keepdim=True)  # [1, 1, dim] type: torch.Tensor
                    query_node_repre_expand = query_node_repre.expand(doc_node_repre.size()[0], -1, -1)  # [node_num, 1, dim] type: torch.Tensor
                    scores = torch.bmm(doc_node_repre, query_node_repre_expand.transpose(1, 2))  # [node_num, sent_len, 1] type: torch.Tensor
                    scores = F.softmax(scores, dim=1)
                    doc_node_repre = torch.mean(doc_node_repre * scores, dim=1)  # [node_num, dim] type: torch.Tensor
                    g.ndata['bart_hidden'][g.filter_nodes(lambda nodes: (nodes.data['type'] != 3))] = doc_node_repre
                    g.ndata['bart_hidden'][g.filter_nodes(lambda nodes: (nodes.data['type'] == 3))] = query_node_repre.squeeze(dim=1)
                    del g.ndata['bart_hidden_w_len']

                g.apply_edges(lambda edges: {'type': self.build_edges(edges, encoder.device)})
                g = dgl.to_heterogeneous(g, [str(jj) for jj in range(ALL_NODE_TYPE)], edge_list, 'type', 'type')  # type: dgl.DGLHeteroGraph
                graph_list.append(g)
            graph = dgl.batch(graph_list)
        return graph

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        addi_source=None,
        addi_source_attention_mask=None,
        addi_source_encoder_outputs=None,
        node_txt_attention_mask=None,
        node_txt_ids=None,
        node_type=None,
        source_node_id=None,
        target_node_id=None,
        graph_hidden_states: torch.Tensor=None,
        graph_hidden_states_mask: torch.Tensor=None,
        batch_idx: torch.Tensor=None,
        node_scores=None,
    ) :
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        score_predict_loss = None
        if graph_hidden_states is None:
            batch_idx_list = batch_idx.int().tolist()
            source_node_id = itemgetter(*batch_idx_list)(source_node_id)
            target_node_id = itemgetter(*batch_idx_list)(target_node_id)
            node_txt_ids = itemgetter(*batch_idx_list)(node_txt_ids)
            node_txt_attention_mask = itemgetter(*batch_idx_list)(node_txt_attention_mask)
            node_type = itemgetter(*batch_idx_list)(node_type)
            node_scores = itemgetter(*batch_idx_list)(node_scores)
            if len(batch_idx_list) == 1:
                source_node_id, target_node_id, node_txt_ids, node_txt_attention_mask, node_type, node_scores = [source_node_id], [target_node_id], [node_txt_ids], [node_txt_attention_mask], [node_type], [node_scores]
            graph = self.build_graph(source_node_id, target_node_id, node_txt_ids, node_txt_attention_mask, node_type, node_scores)
            graph_hidden_states, graph_hidden_states_mask, score_predict_loss = self.magic_graph(graph)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            addi_source=addi_source,
            addi_source_attention_mask=addi_source_attention_mask,
            addi_source_encoder_outputs=addi_source_encoder_outputs,
            graph_hidden_states=graph_hidden_states,
            graph_hidden_states_mask=graph_hidden_states_mask,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        if score_predict_loss is not None and model_args.add_graph_loss > 0:
            masked_lm_loss += score_predict_loss * model_args.add_graph_loss

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def _prepare_encoder_decoder_kwargs_for_generation(self, input_ids: torch.LongTensor, model_kwargs) -> Dict[str, Any]:
        # retrieve encoder hidden states
        encoder = self.get_encoder()
        encoder_kwargs = {
            argument: value for argument, value in model_kwargs.items() if not argument.startswith("decoder_") and not 'node' in argument
        }
        addi_source = model_kwargs['addi_source']
        addi_source_attention_mask = model_kwargs['addi_source_attention_mask']
        del encoder_kwargs['addi_source_attention_mask']
        del encoder_kwargs['addi_source']
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(input_ids, return_dict=True, **encoder_kwargs)
        model_kwargs["addi_source_encoder_outputs"]: ModelOutput = encoder(addi_source, attention_mask=addi_source_attention_mask, return_dict=True, output_attentions=model_kwargs['output_attentions'], output_hidden_states=model_kwargs['output_hidden_states'])

        source_node_id = model_kwargs['source_node_id']
        target_node_id = model_kwargs['target_node_id']
        node_txt_ids = model_kwargs['node_txt_ids']
        node_txt_attention_mask = model_kwargs['node_txt_attention_mask']
        node_type = model_kwargs['node_type']
        node_scores = model_kwargs['node_scores']

        graph = self.build_graph(source_node_id, target_node_id, node_txt_ids, node_txt_attention_mask, node_type, node_scores)
        graph_hidden_states, graph_hidden_states_mask, _ = self.magic_graph(graph)
        model_kwargs["graph_hidden_states"]: torch.Tensor = graph_hidden_states
        model_kwargs["graph_hidden_states_mask"]: torch.Tensor = graph_hidden_states_mask

        return model_kwargs

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs: ModelOutput = None,
        addi_source_encoder_outputs: ModelOutput = None,
        graph_hidden_states: torch.Tensor = None,
        graph_hidden_states_mask: torch.Tensor = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            assert encoder_outputs is not None
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
            )
            model_kwargs["encoder_outputs"] = encoder_outputs
            addi_source_encoder_outputs["last_hidden_state"] = addi_source_encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx.to(addi_source_encoder_outputs.last_hidden_state.device)
            )
            model_kwargs['addi_source_encoder_outputs'] = addi_source_encoder_outputs
            model_kwargs['graph_hidden_states'] = graph_hidden_states.index_select(0, expanded_return_idx)
            model_kwargs['graph_hidden_states_mask'] = graph_hidden_states_mask.index_select(0, expanded_return_idx)
        return input_ids, model_kwargs


    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "addi_source_encoder_outputs": kwargs['addi_source_encoder_outputs'],
            "graph_hidden_states": kwargs['graph_hidden_states'],
            "graph_hidden_states_mask": kwargs['graph_hidden_states_mask'],
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        if cur_len == 1 and self.config.force_bos_token_to_be_generated:
            self._force_token_id_to_be_generated(logits, self.config.bos_token_id)
        elif cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_id_to_be_generated(logits, self.config.eos_token_id)
        return logits

    @staticmethod
    def _force_token_id_to_be_generated(scores, token_id) -> None:
        """force one of token_ids to be generated by setting prob of all other tokens to 0 (logprob=-float("inf"))"""
        scores[:, [x for x in range(scores.shape[1]) if x != token_id]] = -float("inf")

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


@dataclass
class MyDataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    model_args: object = None

    def _add_node_type_list(self, node_type_list: Dict[int, List[int]], node_type: int, node_id: int):
        if node_type in node_type_list:
            node_type_list[node_type].append(node_id)
        else:
            node_type_list[node_type] = [node_id]

    def _add_edge(self, source_node_list: List[int], target_node_list: List[int], src: int, dst: int):
        source_node_list.append(src)
        target_node_list.append(dst)
        if model_args.add_bidirectional_edge:
            source_node_list.append(dst)
            target_node_list.append(src)

    def __call__(self, features):
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )

        all_retrieval = [f['retrieval'] for f in features]
        if 'selftext' in features[0]:
            all_content = [f['title'] + ' ' + f['selftext'] for f in features]
        else:
            all_content = [f['content'] for f in features]

        addi_source_max_length = max([sum(f['addi_source_attention_mask']) for f in features])
        for f in features:
            difference = addi_source_max_length - len(f['addi_source'])
            f['addi_source'] += [self.tokenizer.pad_token_id] * difference
            f['addi_source_attention_mask'] += [0] * difference
            for k in ['adomain', 'qdomain', 'summary', 'token_type_ids', 'retrieval', 'content', 'selftext', 'subreddit', 'answers', 'title']:
                if k in f:
                    del f[k]

        to_return = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batched_node_txt_ids = []
        batched_node_txt_attention_mask = []
        batched_node_scores = []
        batched_source_node_id = []
        batched_target_node_id = []
        batched_node_type = []
        search_indexes = ['search_news', 'search_point', 'search_xueqiu', 'search_qa']
        hit_per_index = 3
        for ex_idx, ex in enumerate(all_retrieval):
            node_id = 1  # 0号预留给当前query
            node_txt = []
            node_txt_ids = []
            node_txt_attention_mask = []
            node_type = []
            node_score = []
            source_node_id = [0]
            target_node_id = [0]  # 加上query节点的self loop
            node_type_list = {}
            self._add_node_type_list(node_type_list, node_type_mapping['query'], 0)

            node_txt.append(all_content[ex_idx].replace('{', '').replace('}', '').replace(' ', ''))
            node_type.append(node_type_mapping['query'])
            node_score.append(0)
            for src_idx, source in enumerate(search_indexes):
                # start = 0 if source != 'search_qa' else 1
                start = 0
                end = start + hit_per_index
                # 在处理qa数据时，如果没有QA数据，或者需要移除QA节点，则用CLS代替
                if source == 'search_qa' and (len(ex[source]) == 0 or model_args.remove_qa_node):
                    node_txt.append(self.tokenizer.cls_token)  # doc node内容是空，就用一个CLS占住
                    node_type.append(node_type_mapping[source])
                    node_score.append(0)
                    self._add_edge(source_node_id, target_node_id,  0, node_id)
                    self._add_node_type_list(node_type_list, node_type_mapping[source], node_id)
                    node_id += 1
                    node_txt.append(self.tokenizer.cls_token)
                    node_type.append(node_type_mapping[source + '_answer'])
                    node_score.append(0)
                    self._add_edge(source_node_id, target_node_id,  node_id - 1, node_id)
                    self._add_node_type_list(node_type_list, node_type_mapping[source + '_answer'], node_id)
                    node_id += 1
                    continue
                if source != 'search_qa' and model_args.remove_doc_node:
                    node_txt.append(self.tokenizer.cls_token)  # doc node内容是空，就用一个CLS占住
                    node_type.append(node_type_mapping[source])
                    node_score.append(0)
                    self._add_edge(source_node_id, target_node_id, 0, node_id)
                    self._add_node_type_list(node_type_list, node_type_mapping[source], node_id)
                    node_id += 1
                    continue
                if source not in ex: continue
                for doc_rank, doc in enumerate(ex[source][start:end]):  # 每一类结果只保留3个
                    doc_node_id = node_id
                    if model_args.expand_sentence_node and source != 'search_qa':  # 展开句子节点
                        node_txt.append(self.tokenizer.cls_token)  # doc node内容是空，就用一个CLS占住
                        node_type.append(node_type_mapping[source])
                        node_score.append(doc_rank)  # doc['score']
                        self._add_edge(source_node_id, target_node_id, 0, node_id)
                        self._add_node_type_list(node_type_list, node_type_mapping[source], node_id)
                        node_id += 1
                        for sent in doc['content']:
                            node_txt.append(sent)
                            node_type.append(node_type_mapping[source+"_sent"])
                            node_score.append(doc_rank)  # doc['score']
                            self._add_edge(source_node_id, target_node_id, doc_node_id, node_id)
                            self._add_node_type_list(node_type_list, node_type_mapping[source+"_sent"], node_id)
                            node_id += 1
                    else:  # 不展开句子节点
                        if source != 'search_qa' and model_args.remove_q_node:
                            node_txt.append(self.tokenizer.cls_token)
                        else:
                            node_txt.append(''.join(doc['content']) if source != 'search_qa' else doc['question'])
                        node_type.append(node_type_mapping[source])
                        node_score.append(doc_rank)  # doc['score']
                        self._add_edge(source_node_id, target_node_id, 0, node_id)
                        self._add_node_type_list(node_type_list, node_type_mapping[source], node_id)
                        node_id += 1
                    if source == 'search_qa':  # 添加答案节点
                        node_txt.append(doc['answer'])
                        node_type.append(node_type_mapping[source + '_answer'])
                        node_score.append(doc_rank)  # doc['score']
                        self._add_edge(source_node_id, target_node_id, node_id - 1, node_id)
                        self._add_node_type_list(node_type_list, node_type_mapping[source + '_answer'], node_id)
                        node_id += 1
                    if model_args.add_comment and source == 'search_point':
                        if len(doc['replylist']) == 0:  # 如果没有评论放个占位符
                            doc['replylist'] = [self.tokenizer.cls_token]
                        for reply in doc['replylist']:
                            node_txt.append(reply)
                            node_type.append(node_type_mapping['comment'])
                            node_score.append(doc_rank)
                            self._add_edge(source_node_id, target_node_id, doc_node_id, node_id)
                            self._add_node_type_list(node_type_list, node_type_mapping['comment'], node_id)
                            node_id += 1
            if node_type_mapping['search_qa_answer'] not in node_type_list:
                logger.info('==================================================')
                logger.info(f'{list(node_type_list.keys())}')
                logger.info(ex)
                logger.info('==================================================')
            else:
                for node_id in node_type_list[node_type_mapping['search_qa_answer']]:
                    self._add_edge(source_node_id, target_node_id, 0, node_id)

            txt_token_result = self.tokenizer(node_txt, max_length=int(self.max_length / 2), padding='max_length',
                                              truncation=True, add_special_tokens=False)
            node_txt_ids.extend(txt_token_result['input_ids'])
            node_txt_attention_mask.extend(txt_token_result['attention_mask'])

            # node_txt_ids[0] = to_return['input_ids'][ex_idx][:int(self.max_length / 2)].numpy().tolist()  # TODO: for graph pretrainning
            # node_txt_ids[0] = node_txt_ids[0] + [self.tokenizer.pad_token_id] * (int(self.max_length / 2) - len(node_txt_ids[0]))
            batched_target_node_id.append(target_node_id)
            batched_source_node_id.append(source_node_id)
            batched_node_type.append(node_type)
            batched_node_txt_ids.append(node_txt_ids)
            batched_node_txt_attention_mask.append(node_txt_attention_mask)
            batched_node_scores.append(node_score)
        to_return['target_node_id'] = batched_target_node_id
        to_return['source_node_id'] = batched_source_node_id
        to_return['node_type'] = batched_node_type
        to_return['node_txt_ids'] = batched_node_txt_ids
        to_return['node_txt_attention_mask'] = batched_node_txt_attention_mask
        to_return['batch_idx'] = torch.range(0, len(batched_target_node_id) - 1)
        to_return['node_scores'] = batched_node_scores
        return to_return


class MySeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs["labels"]
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "addi_source": inputs['addi_source'],
            "addi_source_attention_mask": inputs['addi_source_attention_mask'],
            "node_txt_attention_mask": inputs['node_txt_attention_mask'],
            "node_txt_ids": inputs['node_txt_ids'],
            "node_type": inputs['node_type'],
            "source_node_id": inputs['source_node_id'],
            "target_node_id": inputs['target_node_id'],
            "node_scores": inputs['node_scores'],
            # "edges": inputs['edges'],
            # "edge_mapping": inputs['edge_mapping'],
            "repetition_penalty": 1.2,
        }

        generated_tokens = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return (loss, generated_tokens, labels)


if __name__ == '__main__':

    path = '/home/gaoshen.gao/pretrain/antbart-ckpt-40000'
    tokenizer = BertTokenizerFast.from_pretrained(path)
    # model = BartForConditionalGeneration.from_pretrained(path)
    model = MyBart.from_pretrained(path)


    TXT = f"周三市场呈现开盘指数小幅高开，盘中银行、券商、房地产等权重板块带动拉升"+tokenizer.eos_token


    input_ids = tokenizer([TXT], return_tensors='pt', add_special_tokens=False)['input_ids']
    print('-------call--------')
    logits = model(input_ids).logits  # type: torch.Tensor
    print(logits.shape)
    print('Greedy --> ', tokenizer.decode(logits[0].softmax(dim=1).argmax(dim=1)))
    print('-------generate--------')

    summary_ids = model.generate(input_ids, num_beams=4, max_length=100, early_stopping=True)
    print(tokenizer.decode(summary_ids[0], clean_up_tokenization_spaces=False, skip_special_tokens=True))



