## Introduction

This is a code for our paper ***[Learning to Retrieve Entity-Aware Knowledge and Generate Responses with Copy Mechanism for Task-Oriented Dialogue Systems](https://arxiv.org/abs/2012.11937)*** accepted by AAAI 2021, Workshop on [DSTC 9](https://dstc9.dstc.community).

We modify the code based on [organizer](https://github.com/alexa/alexa-with-dstc9-track1-dataset).

## Requires

- python 3 
- requirements.txt 

## TODO

- [ ] Subtask-1: Knowledge-seeking turn detection
- [ ] Subtask-2: Knowledge selection
- [x] Subtask-3: Knowledge-grounded response generation

## Getting started

* Cd into your working directory.

``` shell
$ cd dstc9-track1
```

* Install the required python packages.

``` shell
$ pip3 install -r requirements.txt
$ python -m nltk.downloader 'punkt'
$ python -m nltk.downloader 'wordnet'
```

## Subtask-3

- Pre-Deal dataset

```shell
bash run_shell/0_dataset_predeal.sh
```

- Train

```shell
bash run_shell/1_train_and_generation.sh train
```

- Generate

```shell
bash run_shell/1_train_and_generation.sh generate
```

## Eval

* Validate the structure and contents of the tracker output.

``` shell
$ python scripts/check_results.py --dataset val --dataroot data/ --outfile baseline_modify_val.json
Found no errors, output file is valid
```

* Evaluate the output.

``` shell
$ python scripts/scores.py --dataset val --dataroot data/ --outfile baseline_modify_val.json --scorefile baseline_modify_val.score.json
```

* Print out the scores.

``` shell
$ cat baseline_modify_val.score.json
{
  "detection": {
    "prec": 0.996268656716418,
    "rec": 0.9988776655443322,
    "f1": 0.9975714552587335
  },
  "selection": {
    "mrr@5": 0.9726010336882744,
    "r@1": 0.9643190734167756,
    "r@5": 0.9815056977395853
  },
  "generation": {
    "bleu-1": 0.43212495074773527,
    "bleu-2": 0.29325946716478263,
    "bleu-3": 0.20037551931972689,
    "bleu-4": 0.14400619504984175,
    "meteor": 0.44184396197275255,
    "rouge_1": 0.47016277347525415,
    "rouge_2": 0.24011047180897968,
    "rouge_l": 0.42534032018695195,
    "generation_bertscore(P, R, F)": [
      0.918968207809873,
      0.9182481457535495,
      0.9185371192263684
    ]
  }
}
```

## Cite

```bibtex
@article{DBLP:journals/corr/abs-2012-11937,
  author    = {Chao{-}Hong Tan and
               Xiaoyu Yang and
               Zi'ou Zheng and
               Tianda Li and
               Yufei Feng and
               Jia{-}Chen Gu and
               Quan Liu and
               Dan Liu and
               Zhen{-}Hua Ling and
               Xiaodan Zhu},
  title     = {Learning to Retrieve Entity-Aware Knowledge and Generate Responses
               with Copy Mechanism for Task-Oriented Dialogue Systems},
  journal   = {CoRR},
  volume    = {abs/2012.11937},
  year      = {2020},
  url       = {https://arxiv.org/abs/2012.11937},
  archivePrefix = {arXiv},
  eprint    = {2012.11937},
  timestamp = {Tue, 05 Jan 2021 16:02:31 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2012-11937.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

