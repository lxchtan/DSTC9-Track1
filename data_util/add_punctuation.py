import os
import sys
import json
import argparse
import nltk


def splitSentence(paragraph):
  tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
  sentences = tokenizer.tokenize(paragraph)
  return sentences

def main(argv):
  parser = argparse.ArgumentParser(description='Evaluate the system outputs.')

  parser.add_argument('--dataset', dest='dataset', action='store', metavar='DATASET', choices=['train', 'val', 'test'],
                      required=True, help='The dataset to analyze')
  parser.add_argument('--dataroot', dest='dataroot', action='store', metavar='PATH', required=True,
                      help='Will look for corpus in <dataroot>/<dataset>/...')
  parser.add_argument("--knowledge", action='store_true', help="Deal with knowledge.")
  parser.add_argument("--knowledge_file", type=str, default="knowledge.json",
                      help="knowledge file name.")
  parser.add_argument('--newDataroot', dest='newDataroot', action='store',metavar='PATH', required=True)

  args = parser.parse_args()

  if not os.path.exists(args.newDataroot):
    os.makedirs(args.newDataroot)
  sub_path = os.path.join(args.newDataroot, args.dataset)  
  if not os.path.exists(sub_path):
    os.mkdir(sub_path)

  # Deal with knowledge
  if args.knowledge:
    with open(os.path.join(args.dataroot, args.knowledge_file), 'r') as f:
      knowledge = json.load(f)
    for domain, contents in knowledge.items():
      for content_id, content in contents.items():
        for doc_id, doc in content['docs'].items():
          if doc['body'][-1] != '.' and doc['body'][-1] != '!':
            doc['body'] += '.'
    with open(os.path.join(args.newDataroot, args.knowledge_file), 'w') as fout:
      json.dump(knowledge, fout, indent=2)
  else: # Deal with logs and labels
    with open(os.path.join(args.dataroot, args.dataset, 'logs.json'), 'r') as f:
      logs = json.load(f)
    with open(os.path.join(args.dataroot, args.dataset, 'labels.json'), 'r') as f:
      labels = json.load(f)

    for label in labels:
      if label['target']:
        response = label['response']
        if len(splitSentence(response)) == 1:
          response = response.replace(" Any other question?", ". Any other question?")\
            .replace(" Anything else",". Anything else").replace(" would you ", ". Would you ")
        if len(splitSentence(response)) == 1:
          label_list = response.split(" ")
          prefix = ['Is', 'Would', 'Do', 'Was', 'Are', 'What', 'Does', 'Can', 'How', 'May', 'Will', 'Where', 'Could', 'Should']
          for p in prefix:
            if p in label_list:
              response = response.replace(' '+p, '. '+p)
        label['response'] = response
    with open(os.path.join(args.newDataroot, args.dataset, 'logs.json'), 'w') as fout:
      json.dump(logs, fout, indent=2)
    with open(os.path.join(args.newDataroot, args.dataset, 'labels.json'), 'w') as fout:
      json.dump(labels, fout, indent=2)

if __name__ == "__main__":
  main(sys.argv)
