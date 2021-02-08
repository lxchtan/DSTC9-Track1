import torch
import os
import json
import random
import logging
import sys

from itertools import chain
from copy import deepcopy

from tqdm import tqdm

from .utils.data import (
  pad_ids, truncate_sequences, pad_input_masks
)

from scripts.dataset_walker import DatasetWalker
from scripts.knowledge_reader import KnowledgeReader

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = {
  "bos_token": "<bos>",
  "eos_token": "<eos>",
  "pad_token": "<pad>",
  "additional_special_tokens": ["<speaker1>", "<speaker2>", "<knowledge_sep>", "<knowledge_tag>"],
}
SPECIAL_TOKENS_VALUES = ["<bos>", "<eos>", "<pad>", "<speaker1>", "<speaker2>", "<knowledge_sep>", "<knowledge_tag>"]


class BaseDataset(torch.utils.data.Dataset):
  def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
    self.args = args
    self.dataroot = args.dataroot
    self.tokenizer = tokenizer
    self.split_type = split_type

    self.SPECIAL_TOKENS = SPECIAL_TOKENS
    self.SPECIAL_TOKENS_VALUES = SPECIAL_TOKENS_VALUES
    self.bos = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["bos_token"])
    self.eos = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["eos_token"])
    self.pad = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["pad_token"])
    self.speaker1, self.speaker2, self.knowledge_sep, self.knowledge_tag = self.tokenizer.convert_tokens_to_ids(
      self.SPECIAL_TOKENS["additional_special_tokens"]
    )
    self.knowledge_sep_token = self.SPECIAL_TOKENS["additional_special_tokens"][2]

    self.all_response_tokenized = []
    self.dataset_walker = DatasetWalker(split_type, labels=labels, dataroot=self.dataroot, labels_file=labels_file)
    self.dialogs = self._prepare_conversations()
    self.all_response_tokenized = list(map(eval, set(map(str, self.all_response_tokenized))))

    self.knowledge_reader = KnowledgeReader(self.dataroot, args.knowledge_file)
    self.knowledge, self.snippets = self._prepare_knowledge()

    self._create_examples()

  def _prepare_conversations(self):
    logger.info("Tokenize and encode the dialog data")
    tokenized_dialogs = []
    for i, (log, label) in enumerate(tqdm(self.dataset_walker, disable=self.args.local_rank not in [-1, 0])): # only show progress bar in one process
      dialog = {}
      dialog["id"] = i
      dialog["log"] = log
      if label is not None:
        if "response" in label:
          label["response_tokenized"] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(label["response"])
          )
          self.all_response_tokenized.append(label["response_tokenized"])
      dialog["label"] = label
      tokenized_dialogs.append(dialog)
    return tokenized_dialogs

  def _prepare_knowledge(self):
    knowledge = self.knowledge_reader.knowledge
    self.knowledge_docs = self.knowledge_reader.get_doc_list()

    tokenized_snippets = dict()
    for snippet in self.knowledge_docs:
      key = "{}__{}__{}".format(snippet["domain"], str(snippet["entity_id"]) or "", snippet["doc_id"])
      knowledge = self._knowledge_to_string(snippet["doc"], name=snippet["entity_name"] or "")
      tokenized_knowledge = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(knowledge))
      tokenized_snippets[key] = tokenized_knowledge[:self.args.knowledge_max_tokens]
    return knowledge, tokenized_snippets

  def _knowledge_to_string(self, doc, name=""):
    return doc["body"]

  def _create_examples(self):
    logger.info("Creating examples")
    self.examples = []
    for dialog in tqdm(self.dialogs, disable=self.args.local_rank not in [-1, 0]):
      dialog_id = dialog["id"]
      label = dialog["label"]
      dialog = dialog["log"]
      if label is None:
        # This will only happen when running knowledge-seeking turn detection on test data
        # So we create dummy target here
        label = {"target": False}

      target = label["target"]

      if not target and self.args.task != "detection":
        # we only care about non-knowledge-seeking turns in turn detection task
        continue

      history = [
        self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(turn["text"]))
        for turn in dialog
      ]
      gt_resp = label.get("response", "")
      tokenized_gt_resp = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(gt_resp))

      # apply history threshold at an utterance-level (a large value can be used to nullify its effect)
      truncated_history = history[-self.args.history_max_utterances:]

      # perform token-level truncation of history from the left
      truncated_history = truncate_sequences(truncated_history, self.args.history_max_tokens)

      if target:
        if "knowledge" not in label:
          # when the labels.json is from knowledge-seeking turn detection,
          # there will be no ground truth knowledge
          # so we just use a dummy snippet here
          if not self.args.eval_all_snippets:
            raise ValueError("eval_all_snippets is required to be true when taking output from knowledge-seeking turn detection")
          label["knowledge"] = [self.knowledge_docs[0]]

        knowledge = label["knowledge"][0]
        knowledge_key = "{}__{}__{}".format(knowledge["domain"], knowledge["entity_id"], knowledge["doc_id"])
        # find snippets with same entity as candidates
        prefix = "{}__{}".format(knowledge["domain"], knowledge["entity_id"])
        knowledge_candidates = [cand for cand in self.snippets.keys() if cand.startswith(prefix)]
        if self.split_type == "train" and self.args.negative_sample_method == "oracle":
          # if there's not enough candidates during training, we just skip this example
          if len(knowledge_candidates) < self.args.n_candidates:
            continue
        used_knowledge = self.snippets[knowledge_key]
        used_knowledge = used_knowledge[:self.args.knowledge_max_tokens]
      else:
        knowledge_candidates = None
        used_knowledge = []

      if target and self.args.__dict__.get("n_response_candidates", 1) > 1:
        response_candidates = self.all_response_tokenized
      else:
        response_candidates = None

      self.examples.append({
        "history": truncated_history,
        "knowledge": used_knowledge,
        "candidates": knowledge_candidates,
        "response": tokenized_gt_resp,
        "response_candidates": response_candidates,
        "response_text": gt_resp,
        "label": label,
        "knowledge_seeking": target,
        "dialog_id": dialog_id
      })

  def build_input_from_segments(self, knowledge, history, response, with_eos=True):
    """ Build a sequence of input from 3 segments: knowledge, history and last reply """
    instance = {}

    sequence = [[self.bos] + knowledge] + history + [response + ([self.eos] if with_eos else [])]
    sequence_with_speaker = [
      [self.speaker1 if (len(sequence) - i) % 2 == 0 else self.speaker2] + s
      for i, s in enumerate(sequence[1:])
    ]
    sequence = [sequence[0]] + sequence_with_speaker
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [self.speaker2 if i % 2 else self.speaker1 for i, s in enumerate(sequence) for _ in s]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]

    return instance, sequence

  def __getitem__(self, index):
    raise NotImplementedError

  def __len__(self):
    return len(self.examples)

class ResponseGenerationDatasetForBertCopy(BaseDataset):
  def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
    super(ResponseGenerationDatasetForBertCopy, self).__init__(args, tokenizer, split_type, labels, labels_file)
    self.punctuate = [".", ",", "!", "?"]
    self.punctuate_token = [self.tokenizer.convert_tokens_to_ids(punc) for punc in self.punctuate]
    self.ignore_knowledge_punctuate = getattr(args, "ignore_knowledge_punctuate", False)
    self.knowledgeROIs_with_input = getattr(args, "knowledgeROIs_with_input", "")
    self.knowledgeROIs_random_mask = getattr(args, "knowledgeROIs_random_mask", False)
    self.input_masks_with_knowledge = getattr(args, "input_masks_with_knowledge", False)
    self.users_ends_with_knowledge = getattr(args, "users_ends_with_knowledge", False)
    self.knowledge_inverse = getattr(args, "knowledge_inverse", False)

  def __getitem__(self, index):
    example = self.examples[index]
    if self.knowledge_inverse:
      example["knowledge"] = example["knowledge"][::-1]
    instance, _ = self.build_input_from_segments(
      example["knowledge"],
      example["history"],
      example["response"]
    )
    return instance

  def build_input_from_segments(self, knowledge, history, response, with_eos=True):
    """ Build a sequence of input from 3 segments: knowledge, history and last reply """
    instance = {}

    if self.users_ends_with_knowledge:
      history[-1] = history[-1] + [self.knowledge_tag] + knowledge

    sequence = [[self.bos] + knowledge] + history + [response + ([self.eos] if with_eos else [])]
    sequence_with_speaker = [
      [self.speaker1 if (len(sequence) - i) % 2 == 0 else self.speaker2] + s
      for i, s in enumerate(sequence[1:])
    ]
    sequence = [sequence[0]] + sequence_with_speaker
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [self.speaker2 if i % 2 else self.speaker1 for i, s in enumerate(sequence) for _ in s]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]

    input_ids_len = len(instance["input_ids"])
    _b_start = input_ids_len - len(response) - (1 if with_eos else 0)
    my_input_mask = [[1 for _ in range(input_ids_len)] for _ in range(input_ids_len)]

    for i in range(_b_start):
      for j in range(_b_start, input_ids_len):
        my_input_mask[i][j] = 0
    for i in range(_b_start, input_ids_len):
      for j in range(i + 1, input_ids_len):
        my_input_mask[i][j] = 0
    instance["input_masks"] = my_input_mask

    knowledge_end = min(instance["input_ids"].index(self.speaker1), instance["input_ids"].index(self.speaker2)) - 1

    input_masks_with_knowledge = deepcopy(my_input_mask)
    if self.input_masks_with_knowledge:
      for i in range(1, knowledge_end):
        for j in range(knowledge_end, _b_start):
          input_masks_with_knowledge[j][i] = 0
    instance["input_masks_with_knowledge"] = input_masks_with_knowledge

    instance["knowledgeROIs"] = [0] + [1] * knowledge_end + [0] * (input_ids_len - knowledge_end - 1)
    if self.knowledgeROIs_random_mask:
      if random.random() < 0.75:
        mask_num = int(knowledge_end * 0.1) + 1
        mask_id = random.sample(range(1, knowledge_end + 1), mask_num)
        for _id in mask_id:
          instance["knowledgeROIs"][_id] = 0
    if self.ignore_knowledge_punctuate:
      for _it, input_id in enumerate(instance["input_ids"][:knowledge_end + 1]):
        if input_id in self.punctuate_token:
          instance["knowledgeROIs"][_it] = 0

    if self.knowledgeROIs_with_input == 'last_sentence':
      sp1 = list(filter(lambda x: x[1] == self.speaker1, enumerate(instance["input_ids"])))
      sp2 = list(filter(lambda x: x[1] == self.speaker2, enumerate(instance["input_ids"])))
      last_sentence_span = sorted([sp1[-1][0], sp2[-1][0]])
      for _it, input_id in enumerate(instance["input_ids"][last_sentence_span[0]:last_sentence_span[1]], start=last_sentence_span[0]):
        if self.ignore_knowledge_punctuate and input_id in self.punctuate_token:
          continue
        else:
          instance["knowledgeROIs"][_it] = 1

    return instance, sequence

  def collate_fn(self, batch):
    input_ids = [ins["input_ids"] for ins in batch]
    token_type_ids = [ins["token_type_ids"] for ins in batch]
    lm_labels = [ins["lm_labels"] for ins in batch]
    input_masks = [ins["input_masks"] for ins in batch]
    input_masks_with_knowledge = [ins["input_masks_with_knowledge"] for ins in batch]
    knowledgeROIs = [ins["knowledgeROIs"] for ins in batch]

    input_ids = torch.tensor(pad_ids(input_ids, self.pad))
    token_type_ids = torch.tensor(pad_ids(token_type_ids, self.pad))
    lm_labels = torch.tensor(pad_ids(lm_labels, -100))
    knowledgeROIs = torch.tensor(pad_ids(knowledgeROIs, 0))
    input_masks = torch.tensor(pad_input_masks(input_masks, 0))
    input_masks_with_knowledge = torch.tensor(pad_input_masks(input_masks_with_knowledge, 0))

    return input_ids, token_type_ids, lm_labels, input_masks, input_masks_with_knowledge, knowledgeROIs

class ResponseGenerationEvalDatasetForBertCopy(ResponseGenerationDatasetForBertCopy):
  def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
    super(ResponseGenerationEvalDatasetForBertCopy, self).__init__(args, tokenizer, split_type, labels, labels_file)
    self.knowledgeROIs_random_mask = False

  def __getitem__(self, index):
    example = self.examples[index]
    if self.knowledge_inverse:
      example["knowledge"] = example["knowledge"][::-1]
    return example

  def collate_fn(self, batch):
    return batch

class KnowledgeSelectionDataset(BaseDataset):
  def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
    super(KnowledgeSelectionDataset, self).__init__(args, tokenizer, split_type, labels, labels_file)
    if self.args.negative_sample_method not in ["all", "mix", "oracle"]:
      raise ValueError("negative_sample_method must be all, mix, or oracle, got %s" % self.args.negative_sample_method)

  def _knowledge_to_string(self, doc, name=""):
    join_str = " %s " % self.knowledge_sep_token
    return join_str.join([name, doc["title"], doc["body"]])

  def __getitem__(self, index):
    example = self.examples[index]

    this_inst = {
      "dialog_id": example["dialog_id"],
      "input_ids": [],
      "token_type_ids": [],
      "mc_token_ids": []
    }

    if self.split_type != "train":
      # if eval_all_snippets is set, we use all snippets as candidates
      if self.args.eval_all_snippets:
        candidates = list(self.snippets.keys())
      else:
        candidates = example["candidates"]
    else:
      if self.args.negative_sample_method == "all":
        candidates = list(self.snippets.keys())
      elif self.args.negative_sample_method == "mix":
        candidates = example["candidates"] + random.sample(list(self.snippets.keys()), k=len(example["candidates"]))
      elif self.args.negative_sample_method == "oracle":
        candidates = example["candidates"]
      else: # although we have already checked for this, still adding this here to be sure
        raise ValueError("negative_sample_method must be all, mix, or oracle, got %s" % self.args.negative_sample_method)

    candidate_keys = candidates
    this_inst["candidate_keys"] = candidate_keys
    candidates = [self.snippets[cand_key] for cand_key in candidates]

    if self.split_type == "train":
      candidates = self._shrink_label_cands(example["knowledge"], candidates)

    label_idx = candidates.index(example["knowledge"])

    this_inst["label_idx"] = label_idx
    for cand in candidates:
      instance, _ = self.build_input_from_segments(
        cand,
        example["history"]
      )
      this_inst["input_ids"].append(instance["input_ids"])
      this_inst["token_type_ids"].append(instance["token_type_ids"])
      this_inst["mc_token_ids"].append(instance["mc_token_ids"])

    return this_inst

  def build_input_from_segments(self, knowledge, history):
    """ Build a sequence of input from 2 segments: knowledge and history"""
    instance = {}

    sequence = [[self.bos]] + history
    sequence_with_speaker = [
      [self.speaker1 if (len(sequence) - i) % 2 == 0 else self.speaker2] + s
      for i, s in enumerate(sequence[1:])
    ]
    sequence = [sequence[0]] + sequence_with_speaker + [[self.knowledge_tag] + knowledge + [self.eos]]

    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [self.speaker2 if i % 2 else self.speaker1 for i, s in enumerate(sequence[:-1]) for _ in s] + [self.knowledge_tag for _ in sequence[-1]]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1

    return instance, sequence

  def _shrink_label_cands(self, label, candidates):
    shrunk_label_cands = candidates.copy()
    shrunk_label_cands.remove(label)
    shrunk_label_cands = random.sample(shrunk_label_cands, k=self.args.n_candidates-1)
    shrunk_label_cands.append(label)
    random.shuffle(shrunk_label_cands)

    return shrunk_label_cands

  def collate_fn(self, batch):
    input_ids = [ids for ins in batch for ids in ins["input_ids"]]
    token_type_ids = [ids for ins in batch for ids in ins["token_type_ids"]]
    mc_token_ids = [id for ins in batch for id in ins["mc_token_ids"]]
    label_idx = [ins["label_idx"] for ins in batch]

    data_info = {
      "dialog_ids": [ins["dialog_id"] for ins in batch],
      "candidate_keys": [ins["candidate_keys"] for ins in batch]
    }

    batch_size = len(batch)
    n_candidates = len(batch[0]["input_ids"])
    input_ids = torch.tensor(
      pad_ids(input_ids, self.pad)
    ).view(batch_size, n_candidates, -1)

    token_type_ids = torch.tensor(
      pad_ids(token_type_ids, self.pad)
    ).view(batch_size, n_candidates, -1)

    lm_labels = torch.full_like(input_ids, -100)
    mc_token_ids = torch.tensor(mc_token_ids).view(batch_size, n_candidates)
    label_idx = torch.tensor(label_idx)

    return input_ids, token_type_ids, mc_token_ids, lm_labels, label_idx, data_info

class KnowledgeTurnDetectionDataset(BaseDataset):
  def build_input_from_segments(self, history):
    """ Build a sequence of input from history """
    instance = {}

    sequence = [[self.bos]] + history[:-1] + [[self.knowledge_tag] + history[-1] + [self.eos]]
    sequence_with_speaker = [
      [self.speaker1 if (len(sequence) - i) % 2 == 0 else self.speaker2] + s
      for i, s in enumerate(sequence[1:])
    ]
    sequence = [sequence[0]] + sequence_with_speaker

    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [self.speaker2 if i % 2 else self.speaker1 for i, s in enumerate(sequence) for _ in s]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1

    return instance, sequence

  def __getitem__(self, index):
    example = self.examples[index]
    instance, _ = self.build_input_from_segments(example["history"])
    instance["label"] = example["knowledge_seeking"]
    instance["dialog_id"] = example["dialog_id"]
    return instance

  def collate_fn(self, batch):
    input_ids = [ins["input_ids"] for ins in batch]
    token_type_ids = [ins["token_type_ids"] for ins in batch]
    mc_token_ids = [ins["mc_token_ids"] for ins in batch]
    labels = [ins["label"] for ins in batch]

    data_info = {
      "dialog_ids": [ins["dialog_id"] for ins in batch]
    }

    input_ids = torch.tensor(pad_ids(input_ids, self.pad))
    token_type_ids = torch.tensor(pad_ids(token_type_ids, self.pad))
    mc_token_ids = torch.tensor(mc_token_ids)
    lm_labels = torch.full_like(input_ids, -100)
    labels = torch.tensor(labels).float()

    return input_ids, token_type_ids, mc_token_ids, lm_labels, labels, data_info

# ResponseGenerationDataset = ResponseGenerationDatasetForBert
# ResponseGenerationEvalDataset = ResponseGenerationEvalDatasetForBert