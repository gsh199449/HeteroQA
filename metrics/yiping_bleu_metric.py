# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors.
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
""" BLEU metric. """
import os
import re
import shutil
import subprocess
import sys
import uuid

import datasets


_CITATION = """\
@INPROCEEDINGS{Papineni02bleu:a,
    author = {Kishore Papineni and Salim Roukos and Todd Ward and Wei-jing Zhu},
    title = {BLEU: a Method for Automatic Evaluation of Machine Translation},
    booktitle = {},
    year = {2002},
    pages = {311--318}
}
"""

_DESCRIPTION = """\
BLEU (bilingual evaluation understudy) is an algorithm for evaluating the quality of text which has been machine-translated from one natural language to another.
Quality is considered to be the correspondence between a machine's output and that of a human: "the closer a machine translation is to a professional human translation,
the better it is" – this is the central idea behind BLEU. BLEU was one of the first metrics to claim a high correlation with human judgements of quality, and
remains one of the most popular automated and inexpensive metrics.

Scores are calculated for individual translated segments—generally sentences—by comparing them with a set of good quality reference translations.
Those scores are then averaged over the whole corpus to reach an estimate of the translation's overall quality. Intelligibility or grammatical correctness
are not taken into account[citation needed].

BLEU's output is always a number between 0 and 1. This value indicates how similar the candidate text is to the reference texts, with values closer to 1
representing more similar texts. Few human translations will attain a score of 1, since this would indicate that the candidate is identical to one of the
reference translations. For this reason, it is not necessary to attain a score of 1. Because there are more opportunities to match, adding additional
reference translations will increase the BLEU score.
"""

_KWARGS_DESCRIPTION = """
Computes BLEU score of translated segments against one or more references.
Args:
    predictions: list of translations to score.
        Each translation should be tokenized into a list of tokens.
    references: list of lists of references for each translation.
        Each reference should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    smooth: Whether or not to apply Lin et al. 2004 smoothing.
Returns:
    'bleu': bleu score,
    'precisions': geometric mean of n-gram precisions,
    'brevity_penalty': brevity penalty,
    'length_ratio': ratio of lengths,
    'translation_length': translation_length,
    'reference_length': reference_length
Examples:

    >>> predictions = [
    ...     ["hello", "there", "general", "kenobi"],                             # tokenized prediction of the first sample
    ...     ["foo", "bar", "foobar"]                                             # tokenized prediction of the second sample
    ... ]
    >>> references = [
    ...     [["hello", "there", "general", "kenobi"], ["hello", "there", "!"]],  # tokenized references for the first sample (2 references)
    ...     [["foo", "bar", "foobar"]]                                           # tokenized references for the second sample (1 reference)
    ... ]
    >>> bleu = datasets.load_metric("bleu")
    >>> results = bleu.compute(predictions=predictions, references=references)
    >>> print(results["bleu"])
    1.0
"""


# @datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class YiPingBleu(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Sequence(datasets.Value("string", id="sequence"), id="references"),
                }
            ),
            codebase_urls=["https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py"],
            reference_urls=[
                "https://en.wikipedia.org/wiki/BLEU",
                "https://towardsdatascience.com/evaluating-text-output-in-nlp-bleu-at-your-own-risk-e8609665a213",
            ],
        )

    def _compute(self, predictions, references, **kwargs):
        references = [s[0] for s in references]
        def sys_bleu_perl(evaluated_list, reference_list):
            if not os.path.exists('tmp/'):
                os.mkdir('tmp/')
            random_name = str(uuid.uuid4())
            evaluates = './tmp/' + random_name + '_e.txt'
            references = './tmp/' + random_name + '_r.txt'
            with open(evaluates, 'w', encoding='utf8') as f1:
                for item in evaluated_list:
                    f1.writelines(item + '\n')
            with open(references, 'w', encoding='utf8') as f2:
                for item in reference_list:
                    f2.writelines(item + '\n')
            ret = sys_bleu_perl_file(evaluates, references)
            os.remove(evaluates)
            os.remove(references)
            # os.removedirs('./tmp')
            shutil.rmtree('./tmp')
            return ret
        def sys_bleu_perl_file(evaluated_file, reference_file):
            perl_path = kwargs['perl_path']
            command = perl_path + ' %s < %s' % (reference_file, evaluated_file)
            result = os.popen(command).readline().strip()

            BLEU_value_pattern = re.compile(
                'BLEU = (\d+.\d+?), (\d+.\d+?), (\d+.\d+?), (\d+.\d+?), (\d+.\d+?) \(BP=(\d+.\d+?), ratio=(\d+.\d+?), hyp_len=(\d+.\d+?), ref_len=(\d+.\d+?)\)')
            value_entry = BLEU_value_pattern.findall(result)[0]
            names = ['blue', 'blue1', 'blue2', 'blue3', 'blue4', 'bp', 'ratio', 'hyp_len', 'ref_len']
            value_dict = {n: float(v) for n, v in zip(names, value_entry)}
            return value_dict
        vv = sys_bleu_perl(predictions, references)
        return vv