import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, NLLLoss, MSELoss
import torch.nn.functional as F

from copy import deepcopy

from transformers import XLNetPreTrainedModel as PreTrainedModel
from transformers import AutoModel

from transformers import BertPreTrainedModel
from transformers import modeling_roberta
from transformers.modeling_roberta import (
    RobertaModel,
    RobertaLMHead,
    RobertaConfig,
    RobertaEmbeddings,
)

from transformers.modeling_utils import SequenceSummary

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = getattr(modeling_roberta, 'ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP', None)


class LatentModel(RobertaModel):
  """
  This class overrides :class:`~transformers.BertModel`. Please check the
  superclass for the appropriate documentation alongside usage examples.
  """

  config_class = RobertaConfig
  pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
  base_model_prefix = "roberta"

  def __init__(self, config):
    super(RobertaModel, self).__init__(config)        # search start from grandpa

    self.embeddings = RobertaEmbeddings(config)
    self.init_weights()

  def forward(
      self,
      input_ids=None,
      attention_mask=None,
      token_type_ids=None,
      position_ids=None,
      head_mask=None,
      inputs_embeds=None,
      encoder_hidden_states=None,
      encoder_attention_mask=None,
      z_hidden_embeds=None,
  ):
    r"""
Return:
    :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
    last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
        Sequence of hidden-states at the output of the last layer of the model.
    pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
        Last layer hidden-state of the first token of the sequence (classification token)
        further processed by a Linear layer and a Tanh activation function. The Linear
        layer weights are trained from the next sentence prediction (classification)
        objective during pre-training.

        This output is usually *not* a good summary
        of the semantic content of the input, you're often better with averaging or pooling
        the sequence of hidden-states for the whole input sequence.
    hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
        Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
        of shape :obj:`(batch_size, sequence_length, hidden_size)`.

        Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
        Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
        :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

        Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
        heads.

Examples::

    from transformers import BertModel, BertTokenizer
    import torch

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)

    last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """

    if input_ids is not None and inputs_embeds is not None:
      raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
      input_shape = input_ids.size()
    elif inputs_embeds is not None:
      input_shape = inputs_embeds.size()[:-1]
    else:
      raise ValueError("You have to specify either input_ids or inputs_embeds")

    device = input_ids.device if input_ids is not None else inputs_embeds.device

    if attention_mask is None:
      attention_mask = torch.ones(input_shape, device=device)
    if token_type_ids is None:
      token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
      extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
      # Provided a padding mask of dimensions [batch_size, seq_length]
      # - if the model is a decoder, apply a causal mask in addition to the padding mask
      # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
      if self.config.is_decoder:
        batch_size, seq_length = input_shape
        seq_ids = torch.arange(seq_length, device=device)
        causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
        causal_mask = causal_mask.to(
            attention_mask.dtype
        )  # causal and attention masks must have same type with pytorch version < 1.3
        extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
      else:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
      raise ValueError(
          "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
              input_shape, attention_mask.shape
          )
      )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    # If a 2D ou 3D attention mask is provided for the cross-attention
    # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
    if self.config.is_decoder and encoder_hidden_states is not None:
      encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
      encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
      if encoder_attention_mask is None:
        encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

      if encoder_attention_mask.dim() == 3:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
      elif encoder_attention_mask.dim() == 2:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
      else:
        raise ValueError(
            "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                encoder_hidden_shape, encoder_attention_mask.shape
            )
        )

      encoder_extended_attention_mask = encoder_extended_attention_mask.to(
          dtype=next(self.parameters()).dtype
      )  # fp16 compatibility
      encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
    else:
      encoder_extended_attention_mask = None

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
    # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
    if head_mask is not None:
      if head_mask.dim() == 1:
        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
      elif head_mask.dim() == 2:
        head_mask = (
            head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        )  # We can specify head_mask for each layer
      head_mask = head_mask.to(
          dtype=next(self.parameters()).dtype
      )  # switch to fload if need + fp16 compatibility
    else:
      head_mask = [None] * self.config.num_hidden_layers

    embedding_output = self.embeddings(
        input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
    )

    if z_hidden_embeds is not None:
      embedding_output = torch.cat([z_hidden_embeds.unsqueeze(1), embedding_output[:, 1:, :]], dim=1)

    encoder_outputs = self.encoder(
        embedding_output,
        attention_mask=extended_attention_mask,
        head_mask=head_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_extended_attention_mask,
    )
    sequence_output = encoder_outputs[0]
    pooled_output = self.pooler(sequence_output)

    outputs = (sequence_output, pooled_output,) + encoder_outputs[
        1:
    ]  # add hidden_states and attentions if they are here
    return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

  def get_input_embeddings(self):
    return self.embeddings.word_embeddings

  def set_input_embeddings(self, value):
    self.embeddings.word_embeddings = value


class LatentCopyGenerationModel(BertPreTrainedModel):
  config_class = RobertaConfig
  pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
  base_model_prefix = "roberta"

  def __init__(self, config):
    config.output_attentions = True
    config.output_hidden_states = True
    super().__init__(config)

    z_summary_config = self.top_config(config)
    # TODO: Find a more suitable way to deal with it.
    cls_summary_config = deepcopy(z_summary_config)
    cls_summary_config.num_labels = 50271 # Roberta + sp_token

    self.roberta = LatentModel(config)
    self.lm_head = RobertaLMHead(config)
    self.z_head = SequenceSummary(z_summary_config)
    self.cls_head = SequenceSummary(cls_summary_config)

    self.p_gen_head = nn.Sequential(nn.Linear(config.hidden_size * 3, 1), nn.Sigmoid())
    self.sample_z = nn.Linear(z_summary_config.num_labels, config.hidden_size, bias=False)

    self.model_stage = 0
    self.myeps = 1e-5
    self.dot_token = config.dot_token

    self.init_weights()

  def top_config(self, config):
    z_summary_config = deepcopy(config)
    z_summary_config.num_labels = config.z_hidden_size
    z_summary_config.summary_type = 'first'
    z_summary_config.summary_use_proj = True
    z_summary_config.summary_activation = None
    z_summary_config.summary_proj_to_labels = True
    z_summary_config.summary_first_dropout = 0.1
    return z_summary_config

  def get_output_embeddings(self):
    return self.lm_head.decoder

  def forward(
      self,
      input_ids=None,
      attention_mask=None,
      token_type_ids=None,
      position_ids=None,
      head_mask=None,
      inputs_embeds=None,
      labels=None,
      z_hidden_embeds=None,
      knowledgeROIs=None,
  ):
    outputs = self.roberta(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        z_hidden_embeds=z_hidden_embeds,
    )
    sequence_output = outputs[0]

    if self.model_stage == 0 or self.model_stage == 1:
      z_logits = self.z_head(sequence_output)
      z_softmax = F.gumbel_softmax(z_logits, tau=0.67, hard=False)
      z = self.sample_z(z_softmax)
      outputs = (z, z_softmax)

    elif self.model_stage == 2:
      prediction_scores = self.lm_head(sequence_output)

      attention_dist = torch.mean(outputs[-1][-1], dim=1)

      p_gen = 1.0
      # select knowledge Areas
      if knowledgeROIs is not None:
        attention_dist *= knowledgeROIs.unsqueeze(-2)
        attention_dist = attention_dist / torch.sum(attention_dist, dim=-1, keepdim=True)

        kRu = knowledgeROIs.unsqueeze(-1)
        knowledge_mean = torch.sum(sequence_output * kRu, dim=1, keepdim=True) / kRu.sum(dim=1, keepdim=True)
        # knowledge_mean = z_hidden_embeds.unsqueeze(1)   # Use z as knowledge mean
        p_gen_input = torch.cat([sequence_output * knowledge_mean, sequence_output,
                                  knowledge_mean.repeat(1, sequence_output.size(1), 1)], dim=-1)

        p_gen = self.p_gen_head(p_gen_input)

      outputs = (prediction_scores, attention_dist, p_gen) + outputs[2:]  # Add hidden states and attention if they are here

      if labels is not None:
        # cal bow loss
        cls_logits = self.cls_head(sequence_output).unsqueeze(1)
        bow_logits = cls_logits.repeat((1, labels.size(-1), 1))
        loss_bl = CrossEntropyLoss()
        bow_loss = loss_bl(bow_logits.view(-1, bow_logits.size(-1)), labels.view(-1))

        # add prediction dict
        attention_dist *= (1 - p_gen)
        prediction_dist = F.softmax(prediction_scores, dim=-1) * p_gen
        prediction_dist = prediction_dist.scatter_add(2, input_ids.unsqueeze(-2).repeat(1, input_ids.size(-1), 1),
                                                      attention_dist)
        # Shift so that tokens < n predict n
        shift_logits = prediction_dist[..., :-1, :].contiguous()
        shift_logits = (shift_logits + self.myeps).log()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        # loss_fct = CrossEntropyLoss()
        loss_fct = NLLLoss()
        masked_lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        norm_loss = self.cal_norm_loss(p_gen, shift_labels)

        outputs = (masked_lm_loss, bow_loss, norm_loss) + outputs
    return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)

  def cal_norm_loss(self, p_gen, shift_labels):
    # Norm loss only force on first sentence(Knowledge part) of a response.
    roll = torch.cat([shift_labels, shift_labels[:, :1]], dim=-1)
    index_ = (roll == self.dot_token).nonzero()
    lambda_norm = 0.6 # Larger for more copy.

    # TODO: Get lambda_ by a more suitable way. Eg: loading from dataset
    lambda_ = torch.zeros_like(roll, dtype=torch.float)
    for i in index_:
      lambda_[i[0], :i[1]] = 1
    lambda_ *= lambda_norm
    labels_mask = (roll != -100)

    loss_norm = MSELoss(reduction='none')
    norm_loss = loss_norm(p_gen, torch.zeros_like(p_gen)).squeeze(-1)
    norm_loss = (norm_loss * labels_mask * lambda_).sum() / labels_mask.sum()

    return norm_loss


class ClsDoubleHeadsModel(PreTrainedModel):
  def __init__(self, config):
    super().__init__(config)
    config.num_labels = 1
    config.summary_activation = None
    config.summary_type = 'cls_index'
    config.summary_proj_to_labels = True

    self.transformer = AutoModel.from_config(config)
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    self.cls_head = SequenceSummary(config)

    self.init_weights()

  def get_output_embeddings(self):
    return self.lm_head

  def forward(
      self,
      input_ids=None,
      attention_mask=None,
      token_type_ids=None,
      position_ids=None,
      head_mask=None,
      inputs_embeds=None,
      mc_token_ids=None,
      lm_labels=None,
      labels=None,
  ):

    transformer_outputs = self.transformer(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
    )

    hidden_states = transformer_outputs[0]

    lm_logits = self.lm_head(hidden_states)
    cls_logits = self.cls_head(hidden_states, mc_token_ids).squeeze(-1)

    outputs = (lm_logits, cls_logits) + transformer_outputs[1:]
    if labels is not None:
      loss_fct = BCEWithLogitsLoss()
      loss = loss_fct(cls_logits, labels)
      outputs = (loss,) + outputs
    if lm_labels is not None:
      shift_logits = lm_logits[..., :-1, :].contiguous()
      shift_labels = lm_labels[..., 1:].contiguous()
      loss_fct = CrossEntropyLoss()
      loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
      outputs = (loss,) + outputs

    return outputs  # (lm loss), (mc loss), lm logits, mc logits, presents, (all hidden_states), (attentions)


class DoubleHeadsModel(PreTrainedModel):
  def __init__(self, config):
    super().__init__(config)
    config.num_labels = 1
    config.summary_activation = None
    config.summary_type = 'cls_index'
    config.summary_proj_to_labels = True

    self.transformer = AutoModel.from_config(config)
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    self.multiple_choice_head = SequenceSummary(config)

    self.init_weights()

  def get_output_embeddings(self):
    return self.lm_head

  def forward(
      self,
      input_ids=None,
      past=None,
      attention_mask=None,
      token_type_ids=None,
      position_ids=None,
      head_mask=None,
      inputs_embeds=None,
      mc_token_ids=None,
      labels=None,
      mc_labels=None,
      use_cache=None,
      output_attentions=None,
      output_hidden_states=None,
      **kwargs
  ):
    if "lm_labels" in kwargs:
      labels = kwargs.pop("lm_labels")
    assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

# TODO: find a more simple way to deal with the shape [114-122]
    transformer_outputs = self.transformer(
        input_ids.view((-1, input_ids.shape[2])),
        attention_mask=attention_mask,
        token_type_ids=token_type_ids.view((-1, token_type_ids.shape[2])),
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
    )

    hidden_states = transformer_outputs[0].view((input_ids.shape[0], input_ids.shape[1], input_ids.shape[2], -1))

    lm_logits = self.lm_head(hidden_states)
    mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)

    outputs = (lm_logits, mc_logits) + transformer_outputs[1:]
    if mc_labels is not None:
      loss_fct = CrossEntropyLoss()
      loss = loss_fct(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1))
      outputs = (loss,) + outputs
    if labels is not None:
      shift_logits = lm_logits[..., :-1, :].contiguous()
      shift_labels = labels[..., 1:].contiguous()
      loss_fct = CrossEntropyLoss()
      loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
      outputs = (loss,) + outputs

    return outputs  # (lm loss), (mc loss), lm logits, mc logits, presents, (all hidden_states), (attentions)
