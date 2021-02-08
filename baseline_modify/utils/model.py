import torch
import torch.nn.functional as F
import logging
import math
import numpy as np
from torch.nn import KLDivLoss, MSELoss
from .metrics import ROUGE_list
from .auxiliary import top_filtering

logger = logging.getLogger(__name__)


def run_batch_generation_for_latentCopy(args, model, batch):
  batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
  input_ids, token_type_ids, lm_labels, input_masks, input_masks_with_knowledge, knowledgeROIs = batch
  ori_model = model.module if hasattr(model, "module") else model
  ori_model.model_stage = 0
  model_outputs = model(input_ids=input_ids, token_type_ids=None, labels=lm_labels, attention_mask=None)
  z, z_distribution = model_outputs[:2]
  ori_model.model_stage = 1
  model_outputs = model(input_ids=input_ids, token_type_ids=None, labels=lm_labels, attention_mask=input_masks)
  z_prior, z_prior_distribution = model_outputs[:2]
  ori_model.model_stage = 2
  model_outputs = model(input_ids=input_ids, token_type_ids=None, labels=lm_labels,
                        attention_mask=input_masks_with_knowledge, z_hidden_embeds=z, knowledgeROIs=knowledgeROIs)
  KLDiv_Loss = KLDivLoss(reduction='batchmean')
  kld_loss = KLDiv_Loss(z_prior_distribution.log(), z_distribution) if getattr(args, "latent_modify", '') != 'real' \
    else KLDiv_Loss(z_distribution.log(), z_prior_distribution)

  lm_loss, bow_loss, norm_loss, lm_logits = model_outputs[:4]
  return lm_loss, lm_logits, (bow_loss, norm_loss), kld_loss


def run_batch_generation_eval_for_latentCopy(args, model, batch):
  batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
  input_ids, token_type_ids, lm_labels, input_masks, input_masks_with_knowledge, knowledgeROIs = batch
  ori_model = model.module if hasattr(model, "module") else model
  ori_model.model_stage = 1
  model_outputs = model(input_ids=input_ids, token_type_ids=None, labels=lm_labels, attention_mask=input_masks)
  z_prior, z_prior_distribution = model_outputs[:2]
  ori_model.model_stage = 2
  model_outputs = model(input_ids=input_ids, token_type_ids=None, labels=lm_labels,
                        attention_mask=input_masks_with_knowledge, z_hidden_embeds=z_prior, knowledgeROIs=knowledgeROIs)

  lm_loss, bow_loss, norm_loss, lm_logits = model_outputs[:4]
  return lm_loss, lm_logits, (bow_loss, norm_loss), torch.tensor([])

def run_batch_generation_greedy_sample_for_latentCopy(args, model, batch, dataset):
  special_tokens_ids = args.tokenizer.convert_tokens_to_ids(dataset.SPECIAL_TOKENS_VALUES)
  current_output = []
  another_data = []

  example = batch[0]
  knowledge, history = example["knowledge"], example["history"]
  response_text = example["response_text"]
  dialog_id = example["dialog_id"]

  instance, sequence = dataset.build_input_from_segments(
    knowledge, history, current_output, with_eos=False
  )

  input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
  input_masks = torch.tensor(instance["input_masks"], device=args.device).unsqueeze(0)

  ori_model = model.module if hasattr(model, "module") else model
  ori_model.model_stage = 1
  model_outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=input_masks)
  z_post, z_post_distribution = model_outputs[:2]
  ori_model.model_stage = 2

  for i in range(args.max_length):
    instance, sequence = dataset.build_input_from_segments(
      knowledge, history, current_output, with_eos=False
    )

    input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
    # input_masks = torch.tensor(instance["input_masks"], device=args.device).unsqueeze(0)
    input_masks_with_knowledge = torch.tensor(instance["input_masks_with_knowledge"], device=args.device).unsqueeze(0)
    knowledgeROIs = torch.tensor(instance["knowledgeROIs"], device=args.device).unsqueeze(0)

    model_outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=input_masks_with_knowledge,
                          z_hidden_embeds=z_post, knowledgeROIs=knowledgeROIs)
    logits, attention_dist, p_gen = model_outputs[:3]

    logits = logits[0, -1, :] / args.temperature
    logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
    probs = F.softmax(logits, dim=-1)

    prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
    if i < args.min_length and prev.item() in special_tokens_ids:
      while prev.item() in special_tokens_ids:
        if probs.max().item() == 1:
          logger.warning("Warning: model generating special token with probability 1! Breaking...")
          break
        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)

    if prev.item() in special_tokens_ids:
      break

    if type(p_gen) != float:
      p_gen = p_gen[0, -1, 0]
    # logger.info(p_gen)

    attention_dist = attention_dist[0, -1, :]
    probs *= p_gen
    attention_dist *= (1 - p_gen)
    probs = probs.scatter_add(0, input_ids.squeeze(0), attention_dist)

    prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
    if i < args.min_length and prev.item() in special_tokens_ids:
      while prev.item() in special_tokens_ids:
        if probs.max().item() == 1:
          logger.warning("Warning: model generating special token with probability 1! Breaking...")
          break
        prev = torch.multinomial(probs, num_samples=1)

    if prev.item() in special_tokens_ids:
      break
    current_output.append(prev.item())

    if type(p_gen) != float:
      another_data.append(format(p_gen.item(), ".4f"))

  return (current_output, another_data), response_text, dialog_id


# Auxiliary for Beam Search
def get_initial_values(args, model, dataset, history, knowledge, model_pre=lambda outputs, **kwargs: outputs,
                       prob_postprocess=lambda outputs, probs, **kwargs: (outputs, probs)):
  outputs = ()
  GFM = True  # args.GFM
  current_output = []
  sub_beam_size = args.sub_beam_size
  group_num = args.group_num
  whole_beam_size = sub_beam_size * group_num

  instance, sequence = dataset.build_input_from_segments(
    knowledge, history, current_output, with_eos=False
  )
  input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
  input_masks = torch.tensor(instance["input_masks"], device=args.device).unsqueeze(0)

  model_args = {
    'input_ids': input_ids,
    'token_type_ids': None,
    'attention_mask': input_masks
  }

  outputs = model_pre(outputs, args=args, model=model, model_args=model_args, instance=instance,
                      whole_beam_size=whole_beam_size)

  model_outputs = model(**model_args)
  logits = model_outputs[0]
  logits = logits[0, -1, :] / args.temperature
  probs = F.softmax(logits, dim=-1)

  outputs, probs = prob_postprocess(outputs, probs, input_ids=input_ids, model_outputs=model_outputs,
                                    whole_beam_size=whole_beam_size)

  _indices = torch.topk(probs, whole_beam_size)[1] \
    if args.no_sample else torch.multinomial(probs, whole_beam_size)  # torch.multinomial(probs, whole_beam_size)
  _values = torch.index_select(torch.log(probs), 0, index=_indices)
  if GFM:
    _index = []
    for i in range(group_num):
      _index.extend([i + group_num * j for j in range(sub_beam_size)])
    _values = _values[_index]
    _indices = _indices[_index]
  score = _values.unsqueeze(-1)
  current_output = _indices.unsqueeze(-1)
  outputs = (current_output, score) + outputs
  return outputs  # (current_output, score, z_post, p_gen_tensors)


def build_inputs(args, current_output, dataset, history, knowledge, whole_beam_size):
  input_ids = []
  input_masks = []
  current_output_list = current_output.cpu().numpy().tolist()

  for j in range(whole_beam_size):
    instance, sequence = dataset.build_input_from_segments(
      knowledge, history, current_output_list[j], with_eos=False
    )
    input_ids.append(torch.tensor(instance["input_ids"], device=args.device))
    input_masks.append(torch.tensor(instance["input_masks"], device=args.device))

  input_ids = torch.stack(input_ids, dim=0)
  input_masks = torch.stack(input_masks, dim=0)
  output = (input_ids, input_masks)

  return output


def cal_next_word(args, score, probs, current_output, indices_shift, special_tokens_ids,
                  place_hold_index, final_output, final_score, finish_index, finish_output_sign):
  sub_beam_size = args.sub_beam_size
  group_num = args.group_num
  tmp_new_scores = torch.log(probs)
  tmp_score = score.repeat((1, tmp_new_scores.size(-1))) + tmp_new_scores
  tmp_score = tmp_score.reshape((group_num, -1))
  tmp_indices = torch.topk(tmp_score, sub_beam_size)[1] if args.no_sample \
    else torch.multinomial(F.softmax(tmp_score, dim=-1), sub_beam_size)
  tmp_score = tmp_score.gather(dim=-1, index=tmp_indices).view(-1, 1)
  tmp_indices = tmp_indices.view(-1, 1)
  last_indices = tmp_indices // tmp_new_scores.size(-1) + indices_shift
  new_indices = tmp_indices % tmp_new_scores.size(-1)
  tmp_current_output = torch.cat([current_output[last_indices.view(-1)], new_indices], dim=-1)
  gain_finish_sentences(args, tmp_score, tmp_current_output, final_output, final_score, finish_index,
                        finish_output_sign,
                        place_hold_index, special_tokens_ids, new_indices, score, last_indices)
  return tmp_current_output, tmp_score


def gain_finish_sentences(args, tmp_score, tmp_current_output, final_output, final_score, finish_index,
                          finish_output_sign,
                          place_hold_index, special_tokens_ids, new_indices=None, score=None, last_indices=None):
  sub_beam_size = args.sub_beam_size
  for j in finish_index.copy():
    if new_indices is None or new_indices[j] in special_tokens_ids:
      group_id = j // sub_beam_size
      group_start = group_id * sub_beam_size
      finish_output_sign[group_id] -= 1
      if finish_output_sign[group_id] == 0:
        for k in range(group_start, group_start + sub_beam_size):
          finish_index.remove(k)
      # Less than zero since finish_index.copy() will not be deleted at this time
      elif finish_output_sign[group_id] < 0:
        finish_output_sign[group_id] = 0
        continue
      place_hold = group_start + sub_beam_size - 1 - finish_output_sign[group_id]
      place_hold_index.append(place_hold)
      final_score[place_hold] = tmp_score[j].item()
      final_output[place_hold] = tmp_current_output[j].cpu().numpy().tolist()
      if last_indices is not None:
        score[
          last_indices.view(-1)[j]] = -10000  # The score wouldn't be reorder, so we need to used the last indices.


def get_final_response(args, knowledge, final_score, final_output):
  select_method = getattr(args, "response_select_method", "final_score")
  output_index = int(np.argmax(final_score))
  if select_method == "rouge_score":
    metric = ROUGE_list()
    rouge_score = []
    for sentence in final_output:
      metric.update((sentence, knowledge))
      rouge_score.append(metric.compute())
    output_index = int(np.argmax(rouge_score))
  real_output = final_output[output_index]
  return real_output

def run_batch_generation_beam_sample_for_latentCopy(args, model, batch, dataset):
  def prob_postprocess_latentCopy(outputs, probs, **kwargs):
    input_ids = kwargs.get('input_ids')
    model_outputs = kwargs.get('model_outputs')
    whole_beam_size = kwargs.get('whole_beam_size')

    attention_dist, p_gen = model_outputs[1:3]
    if not isinstance(p_gen, float):
      p_gen = p_gen[0, -1, 0]
    attention_dist = attention_dist[0, -1, :]
    probs *= p_gen
    attention_dist *= (1 - p_gen)
    probs = probs.scatter_add(0, input_ids.squeeze(0), attention_dist)
    p_gen_tensors = p_gen.repeat((whole_beam_size, 1))
    outputs += (p_gen_tensors,)
    return outputs, probs

  def model_pre_latentCopy(outputs, **kwargs):
    args = kwargs.get('args')
    model = kwargs.get('model')
    model_args = kwargs.get('model_args')
    instance = kwargs.get('instance')
    whole_beam_size = kwargs.get('whole_beam_size')

    model.model_stage = 1
    model_outputs = model(**model_args)
    z_post, z_post_distribution = model_outputs[:2]
    model.model_stage = 2
    input_masks_with_knowledge = torch.tensor(instance["input_masks_with_knowledge"], device=args.device).unsqueeze(0)
    knowledgeROIs = torch.tensor(instance["knowledgeROIs"], device=args.device).unsqueeze(0)
    model_args.update({
      'attention_mask': input_masks_with_knowledge,
      'z_hidden_embeds': z_post,
      'knowledgeROIs': knowledgeROIs
    })

    z_post = z_post.expand((whole_beam_size,) + z_post.size()[1:])
    return outputs + (z_post,)

  def build_inputs_knowledge(args, current_output, dataset, history, knowledge, whole_beam_size):
    input_ids = []
    input_masks = []
    knowledgeROIs = []
    current_output_list = current_output.cpu().numpy().tolist()

    for j in range(whole_beam_size):
      instance, sequence = dataset.build_input_from_segments(
        knowledge, history, current_output_list[j], with_eos=False
      )
      input_ids.append(torch.tensor(instance["input_ids"], device=args.device))
      input_masks.append(torch.tensor(instance["input_masks_with_knowledge"], device=args.device))
      knowledgeROIs.append(torch.tensor(instance["knowledgeROIs"], device=args.device))

    input_ids = torch.stack(input_ids, dim=0)
    input_masks = torch.stack(input_masks, dim=0)
    knowledgeROIs = torch.stack(knowledgeROIs, dim=0)
    return input_ids, input_masks, knowledgeROIs

  build_inputs = build_inputs_knowledge

  # Initial
  sub_beam_size = args.sub_beam_size
  group_num = args.group_num
  whole_beam_size = sub_beam_size * group_num
  special_tokens_ids = args.tokenizer.convert_tokens_to_ids(dataset.SPECIAL_TOKENS_VALUES)
  finish_index = [i for i in range(whole_beam_size)]
  finish_output_sign = [sub_beam_size] * group_num
  final_score = [-1] * whole_beam_size
  final_output = [None] * whole_beam_size
  place_hold_index = []
  indices_shift = torch.tensor(range(0, whole_beam_size, sub_beam_size), dtype=torch.int64, device=args.device) \
    .unsqueeze(-1).repeat(1, sub_beam_size).view(-1).unsqueeze(-1)

  example = batch[0]
  knowledge, history = example["knowledge"], example["history"]
  response_text = example["response_text"]
  dialog_id = example["dialog_id"]

  current_output, score, z_post, p_gen_tensors = get_initial_values(args, model, dataset, history, knowledge,
                                                                    model_pre=model_pre_latentCopy,
                                                                    prob_postprocess=prob_postprocess_latentCopy)

  for i in range(1, args.max_length):
    input_ids, input_masks_with_knowledge, knowledgeROIs = build_inputs(args, current_output, dataset, history,
                                                                        knowledge, whole_beam_size)

    model_outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=input_masks_with_knowledge,
                          z_hidden_embeds=z_post, knowledgeROIs=knowledgeROIs)
    logits, attention_dist, p_gen = model_outputs[:3]
    logits = logits[:, -1, :] / args.temperature
    probs = F.softmax(logits, dim=-1)

    # Jump out with lm score
    cal_next_word(args, score, probs, current_output, indices_shift, special_tokens_ids,
                  place_hold_index, final_output, final_score, finish_index, finish_output_sign)
    if len(finish_index) == 0: break

    # Real calculation
    if type(p_gen) != float:
      p_gen = p_gen[:, -1, :]
    p_gen_tensors = torch.cat([p_gen_tensors, p_gen], dim=-1)
    attention_dist = attention_dist[:, -1, :]
    probs *= p_gen
    attention_dist *= (1 - p_gen)
    probs = probs.scatter_add(1, input_ids, attention_dist)
    current_output, score = cal_next_word(args, score, probs, current_output, indices_shift, special_tokens_ids,
                                          place_hold_index, final_output, final_score, finish_index, finish_output_sign)
    if len(finish_index) == 0: break
  # Remain
  gain_finish_sentences(args, score, current_output, final_output, final_score, finish_index, finish_output_sign,
                        place_hold_index, special_tokens_ids)
  # End
  real_output = get_final_response(args, knowledge, final_score, final_output)
  return (real_output,
          ("Beam Result", final_output, final_score, p_gen_tensors.cpu().numpy().tolist())), response_text, dialog_id


# TODO: reformat
def run_batch_generation_diversity_beam_sample_for_latentCopy(args, model, batch, dataset):
  GFM = True  # args.GFM
  sub_beam_size = args.sub_beam_size
  group_num = args.group_num
  whole_beam_size = sub_beam_size * group_num
  penalty_lambda = getattr(args, "penalty_lambda", 0.6)

  special_tokens_ids = args.tokenizer.convert_tokens_to_ids(dataset.SPECIAL_TOKENS_VALUES)
  current_output = []

  example = batch[0]
  knowledge, history = example["knowledge"], example["history"]
  response_text = example["response_text"]
  dialog_id = example["dialog_id"]

  # Initial
  indices_shift = torch.tensor(range(0, whole_beam_size, sub_beam_size), dtype=torch.int64, device=args.device) \
    .unsqueeze(-1).repeat(1, sub_beam_size).view(-1).unsqueeze(-1)
  finish_index = [i for i in range(whole_beam_size)]
  place_hold_index = []
  finish_output_sign = [sub_beam_size] * group_num
  final_score = [-1] * whole_beam_size
  final_output = [None] * whole_beam_size

  instance, sequence = dataset.build_input_from_segments(
    knowledge, history, current_output, with_eos=False
  )
  input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
  input_masks = torch.tensor(instance["input_masks"], device=args.device).unsqueeze(0)
  input_masks_with_knowledge = torch.tensor(instance["input_masks_with_knowledge"], device=args.device).unsqueeze(0)
  knowledgeROIs = torch.tensor(instance["knowledgeROIs"], device=args.device).unsqueeze(0)

  model.model_stage = 1
  model_outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=input_masks)
  z_post, z_post_distribution = model_outputs[:2]
  model.model_stage = 2

  model_outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=input_masks_with_knowledge,
                        z_hidden_embeds=z_post, knowledgeROIs=knowledgeROIs)
  logits, attention_dist, p_gen = model_outputs[:3]
  logits = logits[0, -1, :] / args.temperature
  probs = F.softmax(logits, dim=-1)
  assert type(p_gen) != float

  p_gen = p_gen[0, -1, 0]
  attention_dist = attention_dist[0, -1, :]
  probs *= p_gen
  attention_dist *= (1 - p_gen)
  probs = probs.scatter_add(0, input_ids.squeeze(0), attention_dist)
  new_scores = torch.log(probs)

  new_indices_list = []
  for _ in range(group_num):
    sub_new_indices = torch.topk(new_scores, sub_beam_size)[1] if args.no_sample \
      else torch.multinomial(F.softmax(new_scores, dim=-1), sub_beam_size)
    new_indices_list.append(sub_new_indices)
    new_scores[sub_new_indices] -= penalty_lambda
  new_indices = torch.cat(new_indices_list, dim=0)

  scores = new_scores[new_indices].unsqueeze(1)
  current_output = new_indices.unsqueeze(1)
  p_gen_tensors = p_gen.repeat((whole_beam_size, 1))

  z_post = z_post.expand((whole_beam_size,) + z_post.size()[1:])
  for i in range(1, args.max_length):
    # Build input
    input_ids = []
    input_masks_with_knowledge = []
    knowledgeROIs = []
    current_output_list = current_output.cpu().numpy().tolist()
    for j in range(whole_beam_size):
      instance, sequence = dataset.build_input_from_segments(
        knowledge, history, current_output_list[j], with_eos=False
      )
      input_ids.append(torch.tensor(instance["input_ids"], device=args.device))
      input_masks_with_knowledge.append(torch.tensor(instance["input_masks_with_knowledge"], device=args.device))
      knowledgeROIs.append(torch.tensor(instance["knowledgeROIs"], device=args.device))
    input_ids = torch.stack(input_ids, dim=0)
    input_masks_with_knowledge = torch.stack(input_masks_with_knowledge, dim=0)
    knowledgeROIs = torch.stack(knowledgeROIs, dim=0)

    model_outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=input_masks_with_knowledge,
                          z_hidden_embeds=z_post, knowledgeROIs=knowledgeROIs)
    logits, attention_dist, p_gen = model_outputs[:3]

    logits = logits[:, -1, :] / args.temperature
    probs = F.softmax(logits, dim=-1)

    # Jump out with lm score
    tmp_new_scores = torch.log(probs)
    new_indices_list = []
    last_indices_list = []
    for i in range(1, group_num + 1):
      sub_new_scores = scores[(i - 1) * sub_beam_size: i * sub_beam_size, ...] \
                       + tmp_new_scores[(i - 1) * sub_beam_size: i * sub_beam_size, ...]
      sub_indices = torch.topk(sub_new_scores.view(-1), sub_beam_size)[1] if args.no_sample \
        else torch.multinomial(F.softmax(sub_new_scores.view(-1), dim=-1), sub_beam_size)
      sub_last_indices = sub_indices // sub_new_scores.size(-1) + (i - 1) * sub_beam_size
      sub_new_indices = sub_indices % sub_new_scores.size(-1)
      new_indices_list.append(sub_new_indices)
      last_indices_list.append(sub_last_indices)
      tmp_new_scores[i * sub_beam_size:, sub_new_indices] -= penalty_lambda
    new_indices = torch.cat(new_indices_list, dim=0)
    last_indices = torch.cat(last_indices_list, dim=0)
    tmp_scores = scores[last_indices] + tmp_new_scores.gather(dim=1, index=new_indices.unsqueeze(1))
    for j in finish_index.copy():
      if new_indices[j] in special_tokens_ids:
        group_id = j // sub_beam_size
        group_start = group_id * sub_beam_size
        finish_output_sign[group_id] -= 1
        if finish_output_sign[group_id] == 0:
          for k in range(group_start, group_start + sub_beam_size): finish_index.remove(k)
        elif finish_output_sign[group_id] < 0:
          continue
        place_hold = group_start + sub_beam_size - 1 - finish_output_sign[group_id]
        # place_hold_index.append(place_hold)
        final_score[place_hold] = tmp_scores[j].item()
        final_output[place_hold] = current_output[j].cpu().numpy().tolist()
        scores[j] = -10000
    if len(finish_index) == 0: break
    # End with Jump out

    # Cal Real Score
    if type(p_gen) != float:
      p_gen = p_gen[:, -1, :]
    attention_dist = attention_dist[:, -1, :]
    probs *= p_gen
    attention_dist *= (1 - p_gen)
    probs = probs.scatter_add(1, input_ids, attention_dist)
    new_scores = torch.log(probs)
    new_indices_list = []
    last_indices_list = []
    for i in range(1, group_num + 1):
      sub_new_scores = scores[(i - 1) * sub_beam_size: i * sub_beam_size, ...] \
                       + new_scores[(i - 1) * sub_beam_size: i * sub_beam_size, ...]
      sub_indices = torch.topk(sub_new_scores.view(-1), sub_beam_size)[1] if args.no_sample \
        else torch.multinomial(F.softmax(sub_new_scores.view(-1), dim=-1), sub_beam_size)
      sub_last_indices = sub_indices // sub_new_scores.size(-1) + (i - 1) * sub_beam_size
      sub_new_indices = sub_indices % sub_new_scores.size(-1)
      new_indices_list.append(sub_new_indices)
      last_indices_list.append(sub_last_indices)
      new_scores[i * sub_beam_size:, sub_new_indices] -= penalty_lambda
    new_indices = torch.cat(new_indices_list, dim=0)
    last_indices = torch.cat(last_indices_list, dim=0)
    scores = scores[last_indices] + new_scores.gather(dim=1, index=new_indices.unsqueeze(1))
    # Break Out
    for j in finish_index.copy():
      if new_indices[j] in special_tokens_ids:
        group_id = j // sub_beam_size
        group_start = group_id * sub_beam_size
        finish_output_sign[group_id] -= 1
        if finish_output_sign[group_id] == 0:
          for k in range(group_start, group_start + sub_beam_size): finish_index.remove(k)
        elif finish_output_sign[group_id] < 0:
          continue
        place_hold = group_start + sub_beam_size - 1 - finish_output_sign[group_id]
        # place_hold_index.append(place_hold)
        final_score[place_hold] = scores[j].item()
        final_output[place_hold] = current_output[j].cpu().numpy().tolist()
        scores[j] = -10000
    if len(finish_index) == 0:
      break
    # End Break Out
    current_output = torch.cat([current_output[last_indices], new_indices.unsqueeze(1)], dim=-1)
    p_gen_tensors = torch.cat([p_gen_tensors, p_gen], dim=-1)

  # Deal with residue
  for j in finish_index:
    group_id = j // sub_beam_size
    group_start = group_id * sub_beam_size
    finish_output_sign[group_id] -= 1
    place_hold = group_start + sub_beam_size - 1 - finish_output_sign[group_id]
    final_score[place_hold] = scores[j].item()
    final_output[place_hold] = current_output[j].cpu().numpy().tolist()

  output_index = int(np.argmax(final_score))
  select_method = getattr(args, "response_select_method", "final_score")
  if select_method == "rouge_score":
    metric = ROUGE_list()
    rouge_score = []
    for sentence in final_output:
      metric.update((sentence, knowledge))
      rouge_score.append(metric.compute())
    output_index = int(np.argmax(rouge_score))
  real_output = final_output[output_index]
  return (real_output,
          ("Beam Result", final_output, final_score, p_gen_tensors.cpu().numpy().tolist())), response_text, dialog_id



def run_batch_selection_train(args, model, batch):
  batch = tuple(input_tensor.to(args.device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
  input_ids, token_type_ids, mc_token_ids, lm_labels, mc_labels = batch
  model_outputs = model(
    input_ids=input_ids, token_type_ids=token_type_ids,
    mc_token_ids=mc_token_ids, mc_labels=mc_labels
  )
  mc_loss = model_outputs[0]
  lm_logits, mc_logits = model_outputs[1], model_outputs[2]
  return mc_loss, lm_logits, mc_logits, mc_labels


def run_batch_selection_eval(args, model, batch):
  candidates_per_forward = args.max_candidates_per_forward_eval * (
    args.n_gpu if isinstance(model, torch.nn.DataParallel) else 1)
  batch = tuple(input_tensor.to(args.device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
  input_ids, token_type_ids, mc_token_ids, _, mc_labels = batch
  all_mc_logits = []
  for index in range(0, input_ids.size(1), candidates_per_forward):
    model_outputs = model(
      input_ids=input_ids[0, index:index + candidates_per_forward].unsqueeze(1),
      token_type_ids=token_type_ids[0, index:index + candidates_per_forward].unsqueeze(1),
      mc_token_ids=mc_token_ids[0, index:index + candidates_per_forward].unsqueeze(1)
    )
    mc_logits = model_outputs[1]
    all_mc_logits.append(mc_logits.detach())
  all_mc_logits = torch.cat(all_mc_logits, dim=0).unsqueeze(0)
  return torch.tensor(0.0), torch.tensor([]), all_mc_logits, mc_labels


def run_batch_detection(args, model, batch):
  batch = tuple(input_tensor.to(args.device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
  input_ids, token_type_ids, mc_token_ids, lm_labels, labels = batch
  model_outputs = model(
    input_ids=input_ids, token_type_ids=token_type_ids,
    mc_token_ids=mc_token_ids, labels=labels
  )
  cls_loss = model_outputs[0]
  lm_logits, cls_logits = model_outputs[1], model_outputs[2]
  return cls_loss, lm_logits, cls_logits, labels


def run_batch_generation(args, model, batch):
  model_name = f"run_batch_generation_for_{args.model_type}"
  return eval(model_name)(args, model, batch)


def run_batch_generation_sample(args, model, batch, dataset):
  middle_name = "beam" if args.beam_search else "greedy"
  diversity = getattr(args, "diversity_beam_search", False)
  return eval(f"run_batch_generation{'_diversity' if diversity else ''}_{middle_name}_sample_for_{args.model_type}")(
    args, model, batch, dataset)
