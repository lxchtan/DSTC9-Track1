import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
try:
  from torch.utils.tensorboard import SummaryWriter
except ImportError:
  from tensorboardX import SummaryWriter

import argparse
import glob
import logging
import os
import random
import shutil
import json

from typing import Dict, List, Tuple
from argparse import Namespace

import numpy as np

from tqdm import tqdm, trange
from transformers import AutoTokenizer

from .utils.argument import update_additional_params
from .utils.model import run_batch_generation_sample
from .utils.training import get_class_for_generate
from .utils.metrics import (
  UnigramMetric, NGramDiversity,
  CorpusNGramDiversity,
  BLEU, METEOR, ROUGE
)
from .utils.data import write_generation_preds, write_generation_preds_debug
from collections import defaultdict

logger = logging.getLogger(__name__)

def set_seed(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if args.n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)


def evaluate(args, eval_dataset, model, tokenizer, desc="") -> Dict:
  if args.local_rank in [-1, 0]:
    eval_output_dir = args.output_dir
    os.makedirs(eval_output_dir, exist_ok=True)

  args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

  eval_sampler = SequentialSampler(eval_dataset)
  eval_dataloader = DataLoader(
    eval_dataset,
    sampler=eval_sampler,
    batch_size=1, # only support batch_size=1 for sampling right now
    collate_fn=eval_dataset.collate_fn
  )

  # multi-gpu evaluate
  # if args.n_gpu > 1:
  #     model = torch.nn.DataParallel(model)

  metrics = [
    UnigramMetric(metric="recall"),
    UnigramMetric(metric="precision"),
    NGramDiversity(n=1),
    NGramDiversity(n=2),
    NGramDiversity(n=3),
    NGramDiversity(n=4),
    CorpusNGramDiversity(n=1),
    CorpusNGramDiversity(n=2),
    CorpusNGramDiversity(n=3),
    CorpusNGramDiversity(n=4),
    BLEU(),
    METEOR(),
    ROUGE()
  ]

  args.tokenizer = tokenizer
  all_output_texts = []
  another_datas = []
  dialog_ids = []
  do_evaluate = False
  model.eval()
  # FixMe: get outputs from `run_batch_generation_sample`
  for batch in tqdm(eval_dataloader, desc="Evaluating", disable=(args.local_rank not in [-1, 0]) or args.display):
    with torch.no_grad():
      sampled_output_ids, ground_truth, dialog_id = run_batch_generation_sample(args, model, batch, eval_dataset)
      if type(sampled_output_ids) == tuple:
        sampled_output_ids, p_gens = sampled_output_ids
        if args.p_debug:
          if p_gens[0] == 'Beam Result':
            _name, final_output, final_score, p_gens_list = p_gens
            # output_text = list(map(lambda x: tokenizer.decode(x, skip_special_tokens=True), final_output))
            # output_bert_score = list(map(lambda x: bert_score.score(x, ground_truth, lang="en")[2].mean(), output_text))
            # all_score = torch.tensor(final_score).softmax() * output_bert_score
            addition_dict = {f'id_{i}': {
              'text': tokenizer.decode(output_ids, skip_special_tokens=True),
              'score': score,
              'p_gens_with_ids': ' '.join(map(lambda x: f'{tokenizer.decode([x[0]], skip_special_tokens=True)}({x[1]:.4f})', zip(output_ids, pgs)))
            } for i, (output_ids, score, pgs) in enumerate(zip(final_output, final_score, p_gens_list))}
            another_datas.append({'beam_outputs': addition_dict})
          else:
            another_datas.append({'p_gens_with_ids': ' '.join(map(
              lambda x: f'{tokenizer.decode([x[0]], skip_special_tokens=True)}({x[1]})',
                                               zip(sampled_output_ids, p_gens)))})
      sampled_output_text = tokenizer.decode(sampled_output_ids, skip_special_tokens=True)
      if args.display:
        _, final_output, final_score = p_gens
        print(f"dialog_id: {dialog_id}")
        print(f"generate: {sampled_output_text}")
        print(json.dumps({f'id_{i}': {
              'text': tokenizer.decode(output_ids),
              'ids': str(output_ids),
              'score': score
            } for i, (output_ids, score) in enumerate(zip(final_output, final_score))}, indent=2))
        print(f"ground_truth: {ground_truth}")
        print()
      # print(''.join([tokenizer.decode([i], skip_special_tokens=True) for i in sampled_output_ids]))
      # print([tokenizer.decode([i], skip_special_tokens=True) for i in sampled_output_ids])
      all_output_texts.append(sampled_output_text)
      dialog_ids.append(dialog_id)
    if ground_truth.strip() != "":
      do_evaluate = True
      for metric in metrics:
        metric.update((sampled_output_text, ground_truth))
    if args.display:
      if len(dialog_ids) == 5:
        raise RuntimeError
    # break
  if args.output_file:
    if args.p_debug:
      write_generation_preds_debug(eval_dataset.dataset_walker, args.output_file, dialog_ids, all_output_texts, another_datas)
    else:
      write_generation_preds(eval_dataset.dataset_walker, args.output_file, dialog_ids, all_output_texts)

  # Save
  result = dict()
  if do_evaluate and args.local_rank in [-1, 0]:
    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    if args.scorefile is not None:
      with open(output_eval_file, "a") as writer, open(args.scorefile, "a") as sf:
        logger.info("***** Eval results %s *****" % desc)
        writer.write("***** Eval results %s *****\n" % desc)
        for metric in metrics:
          name = metric.name()
          score = metric.compute()
          result[name] = score
          logger.info("  %s = %s", name, str(score))
          writer.write("%s = %s\n" % (name, str(score)))
          sf.write("%s = %s\n" % (name, str(score)))
    else:
      with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results %s *****" % desc)
        writer.write("***** Eval results %s *****\n" % desc)
        for metric in metrics:
          name = metric.name()
          score = metric.compute()
          result[name] = score
          logger.info("  %s = %s", name, str(score))
          writer.write("%s = %s\n" % (name, str(score)))

  return result


def main():
  parser = argparse.ArgumentParser()

  # Required parameters
  parser.add_argument("checkpoint", type=str, help="Saved checkpoint directory")
  parser.add_argument('--generate', action='store_true')
  parser.add_argument('--p_debug', action='store_true')
  parser.add_argument('--display', action='store_true')
  parser.add_argument("--generation_params_file", type=str, default="",
                      help="JSON configuration file for generation-related configurations.")
  parser.add_argument("--dataroot", type=str, default="",
                      help="Path to dataset, will override the path in config.")
  parser.add_argument("--eval_dataset", type=str, default="val",
                      help="Dataset to evaluate on, will load dataset from {dataroot}/{eval_dataset}")
  parser.add_argument("--labels_file", type=str, default=None,
                      help="If set, the labels will be loaded not from the default path, but from this file instead."
                           "This option is useful to take the outputs from the previous task in the pipe-lined evaluation.")
  parser.add_argument("--output_file", type=str, default="", help="Predictions will be written to this file.")
  parser.add_argument("--eval_desc", type=str, default="",
                      help="Optional description to be listed in eval_results.txt")
  parser.add_argument('--scorefile',dest='scorefile',action='store',metavar='JSON_FILE', required=False,
                      help='File containing scores')
  parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device (cuda or cpu)")
  parser.add_argument("--local_rank", type=int, default=-1,
                      help="Local rank for distributed training (-1: not distributed)")
  args = parser.parse_args()

  # Setup logging
  logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
  )

  # load args from params file and update the args Namespace
  args.params_file = os.path.join(args.checkpoint, "params.json")
  with open(args.params_file, "r") as f:
    params = json.load(f)
    args = vars(args)
    update_additional_params(params, args)
    args.update(params)
    if len(args["generation_params_file"]) > 0:
      with open(args["generation_params_file"]) as fg:
        generation_params = json.load(fg)
      args.update(generation_params)
    args = Namespace(**args)

  args.params = params # used for saving checkpoints
  dataset_args = Namespace(**args.dataset_args)
  dataset_args.local_rank = args.local_rank
  dataset_args.task = args.task

  # Setup CUDA, GPU & distributed training
  args.distributed = (args.local_rank != -1)
  if not args.distributed:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method='env://')
  args.n_gpu = torch.cuda.device_count() if not args.distributed else 1
  args.device = device

  # Set seed
  set_seed(args)

  # Load pretrained model and tokenizer
  if args.local_rank not in [-1, 0]:
    torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

  args.output_dir = args.checkpoint
  tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
  GenerationModel, dataset_class = get_class_for_generate(args)

  model = GenerationModel.from_pretrained(args.checkpoint)
  model.to(args.device)

  if args.local_rank == 0:
    torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

  logger.info("Generation parameters %s", args)

  # Evaluation
  result = {}
  if args.local_rank in [-1, 0]:
    eval_dataset = dataset_class(dataset_args, tokenizer, split_type=args.eval_dataset, labels_file=args.labels_file)
    result = evaluate(args, eval_dataset, model, tokenizer, desc=args.eval_desc or "val")

  return result

if __name__ == "__main__":
  main()
