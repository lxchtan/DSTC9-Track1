[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Neural Baseline Modify Models for DSTC9 Track 1

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

* Train the baseline models.

``` shell
$ ./train_baseline_modify.sh
```

* Run the baseline models.

``` shell
$ ./run_baseline_modify.sh
```

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
    "prec": 0.955909596146721,
    "rec": 0.9652076318742986,
    "f1": 0.9605361131794491
  },
  "selection": {
    "mrr@5": 0.835610573343261,
    "r@1": 0.7550260610573343,
    "r@5": 0.9303797468354431
  },
  "generation": {
    "bleu-1": 0.3639722536998787,
    "bleu-2": 0.22303651313913814,
    "bleu-3": 0.13792382402704417,
    "bleu-4": 0.08955815914795544,
    "meteor": 0.36541525739465375,
    "rouge_1": 0.3985700061409165,
    "rouge_2": 0.17493494594790954,
    "rouge_l": 0.3532111298013977
  }
}
```

- XLNet Large

```shell
  "selection": {
    "mrr@5": 0.8672065028543063,
    "r@1": 0.8030528667163067,
    "r@5": 0.9370811615785554
  }
```

- Final Results [PLATO-COPY with Other Methods on first two tasks]

```shell
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

## TODO and FIXME

- [x] Rename the automodel into specific model.

