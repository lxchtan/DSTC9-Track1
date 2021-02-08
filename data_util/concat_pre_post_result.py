import os

import sys
import json
import argparse


def main(argv):
  parser = argparse.ArgumentParser(description='Evaluate the system outputs.')

  parser.add_argument('--prefile', dest='prefile', action='store', metavar='JSON_FILE', required=True,
                      help='File containing output JSON')
  parser.add_argument('--postfile', dest='postfile', action='store', metavar='JSON_FILE', required=True,
                      help='File containing output JSON')

  args = parser.parse_args()

  with open(args.prefile, 'r') as f:
    pre_labels= json.load(f)
  with open(args.postfile, 'r') as f:
    post_labels= json.load(f)

  for pre, post in zip(pre_labels, post_labels):
    if pre['target']:
      pre['response'] += ' ' + post['response']

      for _id in range(len(pre['beam_outputs'])):
        pre['beam_outputs'][f'id_{_id}']['text'] += ' ' + post['beam_outputs'][f'id_{_id}']['text']
        pre['beam_outputs'][f'id_{_id}']['score'] += post['beam_outputs'][f'id_{_id}']['score']

  att_pre = list(filter(lambda x: x.startswith('att'), os.path.split(args.prefile)[-1].split('_')))[0]
  att_post = list(filter(lambda x: x.startswith('att'), os.path.split(args.postfile)[-1].split('_')))[0]

  outfile = args.prefile.replace(att_pre, f'att_{att_pre[3:]}_{att_post[3:]}').replace('_pre_', '_combine_')

  with open(outfile, 'w') as fout:
    json.dump(pre_labels, fout, indent=2)

if __name__ == "__main__":
  main(sys.argv)
