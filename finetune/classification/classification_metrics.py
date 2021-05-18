# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluation metrics for classification tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np
import scipy
import sklearn
import math
from sklearn.metrics import roc_curve, auc, precision_score, precision_recall_curve, average_precision_score

from finetune import scorer
import pandas as pd
import csv


class SentenceLevelScorer(scorer.Scorer):
  """Abstract scorer for classification/regression tasks."""

  __metaclass__ = abc.ABCMeta

  def __init__(self, task, lines, split):
    super(SentenceLevelScorer, self).__init__()
    self._total_loss = 0
    self._true_labels = []
    self._preds = []
    self._logits = []
    self._positive_probability = []
    self._task = task
    self._name = task.name
    self._lines = lines.iloc[:, 2:]
    self._split = split

  def update(self, results):
    super(SentenceLevelScorer, self).update(results)
    self._total_loss += results['loss']
    self._true_labels.append(results['label_ids'] if 'label_ids' in results
                             else results['targets'])
    self._preds.append(results['predictions'])
    self._logits.append(results['logits'])
    positive_logit=float(results['logits'][-1])
    positive_probablility=math.exp(positive_logit)/(1+math.exp(positive_logit))
    self._positive_probability.append(positive_probablility)
    

  def get_loss(self):
    return self._total_loss / len(self._true_labels)


class AccuracyScorer(SentenceLevelScorer):

  def _get_results(self):
    correct, count = 0, 0
    for y_true, pred in zip(self._true_labels, self._preds):
      count += 1
      correct += (1 if y_true == pred else 0)
    
    return [
        ('accuracy', 100.0 * correct / count),
        ('loss', self.get_loss()),
    ]


class AUCScorer(SentenceLevelScorer):

  def _get_results(self):  
    fpr, tpr, thersholds = roc_curve(self._true_labels, self._positive_probability)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(self._true_labels, self._positive_probability, pos_label=1)
    average_precision = average_precision_score(self._true_labels, self._positive_probability)
    
    self._positive_label = 1
    n_correct, n_predicted, n_gold = 0, 0, 0
    for y_true, pred in zip(self._true_labels, self._preds):
      if pred == self._positive_label:
        n_gold += 1
        if pred == self._positive_label:
          n_predicted += 1
          if pred == y_true:
            n_correct += 1
    if n_correct == 0:
      p, r, f1 = 0, 0, 0
    else:
      p = 100.0 * n_correct / n_predicted
      r = 100.0 * n_correct / n_gold
      f1 = 2 * p * r / (p + r)
    
    results=np.column_stack((self._lines, self._positive_probability))
    result=pd.DataFrame(results)
    result.to_csv(r'results/' + self._name + '_test_results.txt', index=False, header=None, sep='\t',quoting=csv.QUOTE_NONNUMERIC)
    print('Writing results to results/' + self._name + '_test_results.txt\n')  
    return [
        ('AUC', roc_auc),
        ('AUPR', average_precision),
        ('Precision', p),
        ('Recall', r),
        ('F1', f1),
        ('Loss', self.get_loss()),
    ]


class F1Scorer(SentenceLevelScorer):
  """Computes F1 for classification tasks."""

  def __init__(self, task, split):
    super(F1Scorer, self).__init__(task, split)
    self._positive_label = 1
    self._task = task
    self._name = task.name
    self._split = split

  def _get_results(self):
    n_correct, n_predicted, n_gold = 0, 0, 0
    for y_true, pred in zip(self._true_labels, self._preds):
      if pred == self._positive_label:
        n_gold += 1
        if pred == self._positive_label:
          n_predicted += 1
          if pred == y_true:
            n_correct += 1
    if n_correct == 0:
      p, r, f1 = 0, 0, 0
    else:
      p = 100.0 * n_correct / n_predicted
      r = 100.0 * n_correct / n_gold
      f1 = 2 * p * r / (p + r)

    return [
        ('precision', p),
        ('recall', r),
        ('f1', f1),
        ('loss', self.get_loss()),
    ]


class MCCScorer(SentenceLevelScorer):

  def _get_results(self):
    return [
        ('mcc', 100 * sklearn.metrics.matthews_corrcoef(
            self._true_labels, self._preds)),
        ('loss', self.get_loss()),
    ]


class RegressionScorer(SentenceLevelScorer):

  def _get_results(self):
    preds = np.array(self._preds).flatten()
    return [
        ('pearson', 100.0 * scipy.stats.pearsonr(
            self._true_labels, preds)[0]),
        ('spearman', 100.0 * scipy.stats.spearmanr(
            self._true_labels, preds)[0]),
        ('mse', np.mean(np.square(np.array(self._true_labels) - self._preds))),
        ('loss', self.get_loss()),
    ]