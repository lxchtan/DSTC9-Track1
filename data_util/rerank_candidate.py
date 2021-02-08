import bert_score
import torch
import nltk
import numpy as np
import Levenshtein
from baseline_modify.utils.metrics import (
  UnigramMetric, NGramDiversity,
  CorpusNGramDiversity,
  BLEU, METEOR, ROUGE
)
from scripts.dataset_walker import DatasetWalker
from scripts.knowledge_reader import KnowledgeReader
import re
import os

import sys
import json
import argparse
from collections import defaultdict
from functools import partial

RE_ART = re.compile(r'\b(a|an|the)\b')
RE_PUNC = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')
PUNC = r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']'

metrics = [
  UnigramMetric(metric="recall"),
  UnigramMetric(metric="precision"),
]

def splitSentence(paragraph):
  tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
  sentences = tokenizer.tokenize(paragraph)
  return sentences

class Metric:
  def __init__(self):
    self.reset()

  def reset(self):
    self.score_list = defaultdict(list)
    self.lm_score = []
    self.selection_r1 = []
    self.bertscore = []
    self.refs = []
    self.hyps = []

  def _recall_at_k(self, ref_knowledge, hyp_knowledge, k=5):
    relevance = self._match(ref_knowledge, hyp_knowledge)[:k]

    if True in relevance:
      result = 1
    else:
      result = 0

    return result

  def cal_bertscore(self):
    self.bertscore = bert_score.score(self.hyps, self.refs, lang="en")

  def update(self, ref_text, hyp_text, lm_score):
    self.refs.append(ref_text.strip())
    self.hyps.append(hyp_text.strip())
    self.lm_score.append(lm_score)

    for metric in metrics:
      name = metric.name()
      metric.update((hyp_text, ref_text))
      self.score_list[name].append(metric.compute())
      metric.reset()

def get_response_and_score_meta(candidate, ver='old'):
  if ver == 'old':
    candidate_list = candidate.split('(')
    candidate_text = '('.join(candidate_list[:-1])
    lm_score = eval(candidate_list[-1][:-1])
    return candidate_text, lm_score
  else:
    return candidate['text'], candidate['score']

def set_response_and_score_meta(candidate, text, ver='old'):
  if ver == 'old':
    candidate_list = candidate.split('(')
    candidate_text = '('.join(candidate_list[:-1])
    lm_score = eval(candidate_list[-1][:-1])
    candidate = text + f'({lm_score})'
  else:
    candidate['text'] = text
  return candidate

def set_zeros_lm_score(lm_score, sub_beam_size, group_size):
  lm_score = lm_score.reshape(-1, group_size, sub_beam_size)
  lm_score *= (lm_score.max(dim=-1, keepdim=True)[0]==lm_score)
  lm_score = lm_score.reshape(-1, group_size * sub_beam_size)

def set_unless_lm(output, beam_size, lm_score, lm_zero, ver='old'):
  item_id = 0
  for pred in output:
    if pred['target']:
      for _id in range(beam_size):
        candidate = pred['beam_outputs'][f'id_{_id}']
        candidate_text, _ = get_response_and_score_meta(candidate, ver)
        if candidate_text[-1].isalnum():
          lm_score.view(-1)[item_id] = lm_zero
        item_id += 1

def main(argv):
  parser = argparse.ArgumentParser(description='Evaluate the system outputs.')

  parser.add_argument('--dataset', dest='dataset', action='store', metavar='DATASET', choices=['train', 'val', 'test'],
                      required=True, help='The dataset to analyze')
  parser.add_argument('--dataroot', dest='dataroot', action='store', metavar='PATH', required=True,
                      help='Will look for corpus in <dataroot>/<dataset>/...')
  parser.add_argument("--knowledge_file", type=str, default="knowledge.json",
                      help="knowledge file name.")
  parser.add_argument("--sub_beam_size", type=int, default=2, help="sub_beam_size")
  parser.add_argument("--group_size", type=int, default=4, help="group_size")
  parser.add_argument('--outfile', dest='outfile', action='store', metavar='JSON_FILE', required=True,
                      help='File containing output JSON')
  parser.add_argument('--get_response_version', type=str, default='new')
  parser.add_argument('--from_combine', action='store_true')
  parser.add_argument('--postfile', type=str, default='')

  args = parser.parse_args()

  with open(args.outfile, 'r') as f:
    output = json.load(f)
  if args.from_combine:
    postfile = args.postfile or re.sub(r'att_(\d+)_(\d+)', lambda m: f'att{m.group(2)}', args.outfile).replace('combine', 'post')
    with open(postfile, 'r') as f:
      post_output = json.load(f)

  knowledge_reader = KnowledgeReader(dataroot=args.dataroot, knowledge_file=args.knowledge_file)
  beam_size = args.sub_beam_size * args.group_size
  version = args.version

  get_response_and_score = partial(get_response_and_score_meta, ver=args.get_response_version)

  med_radio_list = []
  med_score_list = []
  whole_knowledge_list = []

  metric = Metric()
  for pid, pred in enumerate(output):
    if pred['target']:
      front_txt = []
      post_txt = []
      lm_scores = []
      ref_text = knowledge_reader.get_doc(**pred['knowledge'][0])['doc']['body']
      whole_knowledge_list.append(ref_text)
      p_response = pred['response']
      p_response_list = splitSentence(p_response)
      if len(p_response_list) > 1:
        p_response_list = [' '.join(p_response_list[:i]) for i in range(1, len(p_response_list))]
        p_response_list_med = [Levenshtein.distance(ref_text, candidate_text) for candidate_text in p_response_list]
        p_response_front = p_response_list[int(np.argmin(p_response_list_med))]
        p_response_post = p_response[len(p_response_front)+1:].strip()
      for _id in range(beam_size):
        candidate = pred['beam_outputs'][f'id_{_id}']
        candidate_text, lm_score = get_response_and_score(candidate)
        candidate_text_list = splitSentence(candidate_text)
        if not args.from_combine:
          lm_scores.append(lm_score)
        else:
          post_cadidate = post_output[pid]['beam_outputs'][f'id_{_id}']
          _post_t, post_score = get_response_and_score(post_cadidate)
          lm_scores.append(post_score)
        if len(candidate_text_list) > 1:
          candidate_text_list = [' '.join(candidate_text_list[:i]) for i in range(1, len(candidate_text_list))]
          candidate_text_list_med = [Levenshtein.distance(ref_text, candidate_text) for candidate_text in candidate_text_list]
          candidate_text_after = candidate_text_list[int(np.argmin(candidate_text_list_med))]
          front_txt.append(candidate_text_after)
          if args.from_combine:
            post_txt.append(_post_t)
          else:
            post_txt.append(candidate_text[len(candidate_text_after)+1:].strip())
          candidate_text = candidate_text_after
        else:
          front_txt.append(candidate_text)
          post_txt.append(candidate_text)
        dis_func = Levenshtein.jaro_winskler
        med_radio_list.append(dis_func(candidate_text, ref_text))
        metric.update(ref_text, candidate_text, lm_score)

  scores = metric.score_list

  metric.cal_bertscore()
  bert_score = metric.bertscore
  lm_score = metric.lm_score

  bert_score = bert_score[2].reshape((-1, beam_size))
  lm_score = torch.tensor(lm_score).reshape((-1, beam_size))

  med_radio_score = torch.tensor(med_radio_list).reshape((-1, beam_size))
  lm_score = (lm_score - lm_score.min())/(lm_score.max() - lm_score.min())
  set_zeros_lm_score(lm_score, args.sub_beam_size, args.group_size)
  bert_score -= bert_score.min(dim=-1, keepdim=True)[0]
  bert_score /= bert_score.max(dim=-1, keepdim=True)[0]
  med_part = torch.where(med_radio_score > 0.9, med_radio_score, torch.zeros_like(med_radio_score)) * 0.5
  final_score = bert_score + lm_score - med_part
  print(med_radio_score[0])
  print(bert_score[0], lm_score[0], med_part[0])

  select = final_score.argmax(dim=-1)

  item_id = 0
  for pred in output:
    if pred['target']:
      candidate_text, _ = get_response_and_score(pred['beam_outputs'][f'id_{select[item_id].item()}'])
      pred['response'] = candidate_text
      item_id += 1

  with open(os.path.join(args.outfile[:-5] + f'_rerank{version}.json'), 'w') as fout:
    json.dump(output, fout, indent=2)

if __name__ == "__main__":
  main(sys.argv)
