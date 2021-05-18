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

"""Returns task instances given the task name."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import configure_finetuning
from finetune.classification import classification_tasks
from finetune.qa import qa_tasks
from finetune.tagging import tagging_tasks
from model import tokenization


def get_tasks(config: configure_finetuning.FinetuningConfig):
  tokenizer = tokenization.FullTokenizer(vocab_file=config.vocab_file,
                                         do_lower_case=config.do_lower_case)
  return [get_task(config, task_name, tokenizer)
          for task_name in config.task_names]


def get_task(config: configure_finetuning.FinetuningConfig, task_name,
             tokenizer):
  """Get an instance of a task based on its name."""
  if task_name == "ST":
    return classification_tasks.ST(config, tokenizer)
  elif task_name == "CDK":
    return classification_tasks.CDK(config, tokenizer)
  elif task_name == "CK2":
    return classification_tasks.CK2(config, tokenizer)
  elif task_name == "MAPK":
    return classification_tasks.MAPK(config, tokenizer)
  elif task_name == "PKA":
    return classification_tasks.PKA(config, tokenizer)
  elif task_name == "PKC":
    return classification_tasks.PKC(config, tokenizer)
  else:
    raise ValueError("Unknown task " + task_name)
