import nltk
import numpy as np
import Levenshtein

from scripts.knowledge_reader import KnowledgeReader
import re
import os

import sys
import json
import argparse

RE_ART = re.compile(r'\b(a|an|the)\b')
RE_PUNC = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')


def splitSentence(paragraph):
  tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
  sentences = tokenizer.tokenize(paragraph)
  return sentences

def main(argv):
  parser = argparse.ArgumentParser(description='Evaluate the system outputs.')

  parser.add_argument('--dataset', dest='dataset', action='store', metavar='DATASET', choices=['train', 'val', 'test'],
                      default='train', help='The dataset to analyze')
  parser.add_argument('--dataroot', dest='dataroot', action='store', metavar='PATH', default='data_modify/add_stop',
                      help='Will look for corpus in <dataroot>/<dataset>/...')
  parser.add_argument("--knowledge_file", type=str, default="knowledge.json",
                      help="knowledge file name.")

  args = parser.parse_args()

  # data = DatasetWalker(dataroot=args.dataroot, dataset=args.dataset, labels=False)
  knowledge_reader = KnowledgeReader(dataroot=args.dataroot, knowledge_file=args.knowledge_file)
  # beam_size = len(output[0]['beam_outputs'])

  with open(os.path.join(args.dataroot, args.dataset, 'logs.json'), 'r') as f:
    logs = json.load(f)
  with open(os.path.join(args.dataroot, args.dataset, 'labels.json'), 'r') as f:
    labels = json.load(f)

  count_1 = 0
  new_logs = []
  new_labels_pre = []
  new_labels_post = []
  for log, label in zip(logs, labels):
    if label['target']:
      response = label['response']
      ref_text = knowledge_reader.get_doc(**label['knowledge'][0])['doc']['body']
      candidate_text_list = splitSentence(response)
      if len(candidate_text_list) > 1:
        candidate_text_list = [' '.join(candidate_text_list[:i]) for i in range(1, len(candidate_text_list))]
        candidate_text_list_med = [Levenshtein.distance(ref_text, candidate_text) for candidate_text in candidate_text_list]
        candidate_text_after = candidate_text_list[int(np.argmin(candidate_text_list_med))]
        post_txt = response[len(candidate_text_after) + 1:].strip()
        pre_txt = candidate_text_after
        pre_label = label.copy()
        pre_label['response'] = pre_txt
        post_label = label.copy()
        post_label['response'] = post_txt
        new_logs.append(log)
        new_labels_pre.append(pre_label)
        new_labels_post.append(post_label)

  pre_path = os.path.join(args.dataroot, 'pre_response', args.datasest)
  post_path = os.path.join(args.dataroot, 'post_response', args.datasest)
  if not os.path.exists(pre_path):
    os.makedirs(pre_path)
  if not os.path.exists(post_path):
    os.makedirs(post_path)

  with open(os.path.join(pre_path, 'labels.json'), 'w') as fout:
    json.dump(new_labels_pre, fout, indent=2)
  with open(os.path.join(post_path, 'labels.json'), 'w') as fout:
    json.dump(new_labels_post, fout, indent=2)

  with open(os.path.join(pre_path, 'logs.json'), 'w') as fout:
    json.dump(new_logs, fout, indent=2)
  with open(os.path.join(post_path, 'logs.json'), 'w') as fout:
    json.dump(new_logs, fout, indent=2)

if __name__ == "__main__":
  main(sys.argv)
