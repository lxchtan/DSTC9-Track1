import numpy as np

from ..dataset import (
  ResponseGenerationDatasetForBertCopy,
  KnowledgeSelectionDataset,
  KnowledgeTurnDetectionDataset,

  ResponseGenerationEvalDatasetForBertCopy,
)

from ..models import (
  ClsDoubleHeadsModel,
  DoubleHeadsModel,
  LatentCopyGenerationModel,
)

from .model import (
  run_batch_detection,
  run_batch_generation,
  run_batch_generation_eval_for_latentCopy,
  run_batch_selection_train,
  run_batch_selection_eval
)

def get_classes(args):
  task = args.task

  if task.lower() == "generation":
    run_batch_generation_eval = run_batch_generation_eval_for_latentCopy
    return ResponseGenerationDatasetForBertCopy, LatentCopyGenerationModel, run_batch_generation, run_batch_generation_eval

  elif task.lower() == "selection":
    return KnowledgeSelectionDataset, DoubleHeadsModel, run_batch_selection_train, run_batch_selection_eval
  elif task.lower() == "detection":
    return KnowledgeTurnDetectionDataset, ClsDoubleHeadsModel, run_batch_detection, run_batch_detection
  else:
    raise ValueError("args.task not in ['generation', 'selection', 'detection'], got %s" % task)

def get_class_for_generate(args):
  GenerationModel = LatentCopyGenerationModel
  dataset_class = ResponseGenerationEvalDatasetForBertCopy

  return GenerationModel, dataset_class


def regain_loss(args, loss, loss_2, loss_3, train_epoch, percentage):
  kk_loss = []
  if args.task == 'generation':
    loss_1 = loss
    bow_loss, norm_loss = loss_2
    loss_2 = bow_loss + norm_loss
    kk_loss = [loss_1, bow_loss, norm_loss, loss_3]
    loss = loss_1 + loss_2 + loss_3

  multi_loss = np.array([l.detach().cpu().numpy() for l in kk_loss])
  return loss, multi_loss
