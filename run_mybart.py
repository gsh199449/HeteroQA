#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
import json

import logging
import re
import stat
import sys
import time
from data_process_functions import load_function

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
import math
import os
from enum import Enum

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict

import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, load_metric, DatasetDict, DownloadConfig

import transformers
from filelock import FileLock
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed, BertTokenizerFast, BartConfig, BartTokenizer,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

@dataclass
class ArgClassBase:
    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support).
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
        return d

@dataclass
class ModelArguments(ArgClassBase):
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    addi_share_decoder_attn: bool = field(
        default=False,
        metadata={"help": "addi_share_decoder_attn"},
    )
    two_hidden_merge_method: str = field(
        default='cat',
        metadata={"help": "cat/add/weightsum/fusion"},
    )
    expand_sentence_node: bool = field(
        default=False,
        metadata={"help": "expand document nodes to sentences nodes, or use all the sentences in document node"},
    )
    hetero_graph: bool = field(
        default=False,
        metadata={"help": "use heterogeneous graph"},
    )
    hetero_graph_model: str = field(
        default='hgt',
        metadata={"help": "use hgt/han"},
    )
    add_bidirectional_edge: bool = field(
        default=False,
        metadata={"help": "add bidirectional edge between query and document"},
    )
    add_graph_loss: float = field(
        default=0.0,
        metadata={"help": "add graph loss function to final loss"},
    )
    graph_node_residual: bool = field(
        default=False,
        metadata={"help": "add node bart representation to graph representation"},
    )
    graph_layer_num: int = field(
        default=4,
        metadata={"help": "number of graph layers"},
    )
    add_comment: bool = field(
        default=False,
        metadata={"help": "add comment node"},
    )
    remove_qa_node: bool = field(
        default=False,
        metadata={"help": "remove qa nodes"},
    )
    remove_q_node: bool = field(
        default=False,
        metadata={"help": "remove qa nodes"},
    )
    remove_doc_node: bool = field(
        default=False,
        metadata={"help": "remove document nodes"},
    )
    node_interaction_w_query: bool = field(
        default=False,
        metadata={"help": "make interaction for doc node with query node"},
    )
    magic_hgt: bool = field(
        default=False,
        metadata={"help": "use magic hgt"},
    )
    def __post_init__(self):
        if self.two_hidden_merge_method is not None and self.two_hidden_merge_method not in ['cat', 'add', 'weightsum', 'fusion']:
            raise ValueError("two_hidden_merge_method must choose from ['cat', 'add', 'weightsum', 'fusion'].")
        if self.hetero_graph_model == 'han' and not self.add_bidirectional_edge:
            raise ValueError("han model requires bidirectional edge ")


@dataclass
class DataTrainingArguments(ArgClassBase):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    exp_name: str = field(metadata={"help": "The name of the project."})
    proj_name: str = field(
        default="seq2seqV4",
        metadata={"help": "The name of the project."},
    )
    log_root: Optional[str] = field(
        default=os.path.expanduser('./log'), metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    save_dataset_path: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    data_srcs: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    data_tgts: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge/sacreblue) on "
                    "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge/sacreblue) on "
                    "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=200,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=100,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                    "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                    "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                    "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
                    "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
                    "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=4,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    fix_bart_step: Optional[int] = field(
        default=-1,
        metadata={
            "help": "-1 means training bart from start,"
        },
    )
    is_chinese: bool = field(
        default=False,
        metadata={
            "help": "chinese"
        },
    )
    log_gradient: bool = field(
        default=False,
        metadata={
            "help": "log_gradient on tensorboard"
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None and self.save_dataset_path is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    # If we pass only one argument to the script and it's the path to a json file,
    # let's parse it to get our arguments.
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))  # type: ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments
else:
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()  # type: ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments


def main():
    from magic_bart import MyBart, MyDataCollatorForSeq2Seq, MySeq2SeqTrainer
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    training_args.logging_steps = 10
    data_args.log_root = os.path.join(data_args.log_root, data_args.proj_name, data_args.exp_name)
    training_args.output_dir = os.path.join(data_args.log_root, 'model')
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            # logging.warning(f'last train is ended, continue train on {training_args.output_dir}')
            # last_checkpoint = training_args.output_dir
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
            model_args.model_name_or_path = last_checkpoint
    if training_args.do_predict and model_args.model_name_or_path is None:
        model_args.model_name_or_path = get_last_checkpoint(training_args.output_dir)
        logger.info(f'loading checkpoint from {model_args.model_name_or_path}')

    # logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Dataset parameters %s", data_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if not training_args.do_train and (training_args.do_eval or training_args.do_predict) and model_args.model_name_or_path is None:
        # 纯测试且没指定ckpt 就用最新的ckpt
        model_args.model_name_or_path = last_checkpoint if last_checkpoint is not None else training_args.output_dir

    if model_args.model_name_or_path is None:
        logger.info('******* Initializing model form scratch **********')
        if data_args.is_chinese:
            tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
        else:
            tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        tokenizer.add_special_tokens({
            'bos_token': '<s>',
            'eos_token': '</s>',
        })
        config = BartConfig(encoder_layers=6, decoder_layers=6, encoder_ffn_dim=2048, decoder_ffn_dim=2048, encoder_attention_heads=8, decoder_attention_heads=8)
        # config = BartConfig(encoder_layers=3, decoder_layers=3, encoder_ffn_dim=1024, decoder_ffn_dim=1024, encoder_attention_heads=8, decoder_attention_heads=8)
        config.decoder_start_token_id = tokenizer.bos_token_id
        config.pad_token_id = tokenizer.pad_token_id
        config.num_beams = data_args.num_beams
        config.max_length = data_args.max_target_length
        model = MyBart(config)
    else:
        logger.info(f'******* Loading model form pretrained {model_args.model_name_or_path} **********')
        # logger.info('load config')
        # config = BartConfig.from_pretrained(model_args.model_name_or_path)
        if 'fnlp' in model_args.model_name_or_path:
            tokenizer = BertTokenizerFast.from_pretrained(model_args.model_name_or_path)
            model = MyBart.from_pretrained(model_args.model_name_or_path)
        elif 'bart-base' in model_args.model_name_or_path or 'bart-large' in model_args.model_name_or_path:
            logger.info('load tokenizer')
            tokenizer = BartTokenizer.from_pretrained(model_args.model_name_or_path)
            logger.info('load model')
            model = MyBart.from_pretrained(model_args.model_name_or_path)
        else:
            try:
                tokenizer = BertTokenizerFast.from_pretrained(model_args.model_name_or_path)
            except TypeError:
                logger.info('changing to BartTokenizer')
                tokenizer = BartTokenizer.from_pretrained(model_args.model_name_or_path)
            model = MyBart.from_pretrained(model_args.model_name_or_path)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if data_args.save_dataset_path is None:
        logger.info('******* Making Dataset **********')
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        if extension == 'txt': extension = 'text'
        datasets = load_dataset(extension, data_files=data_files, download_config=DownloadConfig(use_etag=False))
        # Temporarily set max_target_length for training.
        max_target_length = data_args.max_target_length
        padding = "max_length" if data_args.pad_to_max_length else False

        if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
            logger.warn(
                "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
                f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
            )

        datasets = datasets.map(
            load_function('graph', tokenizer, data_args, max_target_length, padding),
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            # load_from_cache_file=not data_args.overwrite_cache,
        )

        logger.info('saving dataset')
        data_path = './msm-dataset'
        datasets.save_to_disk(data_path)
        logger.info(f'save dataset finish {data_path}')
        exit(0)
    else:
        logger.info(f'******* Loading Dataset from {data_args.save_dataset_path} **********')
        datasets = DatasetDict.load_from_disk(data_args.save_dataset_path)

    train_dataset = datasets["train"] if training_args.do_train is not None and "train" in datasets else None
    eval_dataset = datasets["validation"] if training_args.do_eval is not None and "validation" in datasets else None
    test_dataset = datasets["test"] if training_args.do_predict is not None and "test" in datasets else datasets["validation"]
    if training_args.do_predict is None and "test" not in datasets:
        logging.warning(f'using validation dataset as test!')

    max_target_length = data_args.val_max_target_length
    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = MyDataCollatorForSeq2Seq(
            tokenizer,
            max_length=data_args.max_source_length,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
            model_args=model_args
        )

    # Metric
    bleu_metric = load_metric('metrics/sacrebleu.py')
    yiping_bleu_metric = load_metric('metrics/yiping_bleu_metric.py')

    def postprocess_text(metric_name, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        if data_args.is_chinese:
            split_char = lambda x: ' '.join(list(x))
        else:
            split_char = lambda x: x

        # rougeLSum expects newline after each sentence
        if metric_name == "rouge":
            preds = ["\n".join([split_char(s) for s in nltk.sent_tokenize(pred)]) for pred in preds]
            labels = ["\n".join([split_char(s) for s in nltk.sent_tokenize(label)]) for label in labels]
        elif metric_name == 'sacrebleu':  # sacrebleu
            labels = [[split_char(label)] for label in labels]
            preds = [split_char(p) for p in preds]
        elif metric_name == 'yiping_bleu':
            labels = [[split_char(label)] for label in labels]
            preds = [split_char(p) for p in preds]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # 获取当前状态，如果是evaluation就用eval数据集，如果是predict就用test数据集
        import traceback
        method_name = [s.name for s in traceback.extract_stack() if s.filename.endswith('trainer_seq2seq.py')]
        if len(method_name) == 0:
            logger.fatal(f'method name is none {method_name}')
        method_name = method_name[0]
        if method_name == 'predict':
            dataset = test_dataset
        else:
            dataset = eval_dataset
        addi_source_str = tokenizer.batch_decode(dataset['addi_source'], skip_special_tokens=True)

        replace_special_token = lambda x: re.sub('\[.*?\]', '', x).replace('\n', '')
        if data_args.is_chinese:
            decoded_preds = [replace_special_token(p.replace(' ', '').strip()) for p in decoded_preds]
            decoded_labels = [replace_special_token(p.replace(' ', '').strip()) for p in decoded_labels]
        else:
            decoded_preds = [replace_special_token(p.strip()) for p in decoded_preds]
            decoded_labels = [replace_special_token(p.strip()) for p in decoded_labels]

        time_str = time.strftime("%d-%H-%M", time.localtime())
        os.makedirs(os.path.join(data_args.log_root, f'decode-{trainer.state.global_step}-{time_str}'))
        fo_ref = open(os.path.join(data_args.log_root, f'decode-{trainer.state.global_step}-{time_str}', 'reference.txt'), 'w',
                      encoding='utf8')
        fo_dec = open(os.path.join(data_args.log_root, f'decode-{trainer.state.global_step}-{time_str}', 'decoded.txt'), 'w',
                      encoding='utf8')
        fo_show = open(os.path.join(data_args.log_root, f'decode-{trainer.state.global_step}-{time_str}', 'show.txt'), 'w',
                      encoding='utf8')
        for pred, lab, inp_str, addi in zip(decoded_preds, decoded_labels, dataset['content'] if 'content' in dataset.column_names else dataset['title'], addi_source_str):
            fo_ref.write(f'{lab}\n')
            fo_dec.write(f'{pred}\n')
            if data_args.is_chinese:
                fo_show.write(f'{inp_str.replace(" ", "")}\n{addi.replace(" ", "")}\n{lab}\n{pred}\n{"-"*20}\n')
            else:
                fo_show.write(f'{inp_str}\n{addi}\n{lab}\n{pred}\n{"-"*20}\n')

        result = {}

        # evaluate bleu
        bleu_decoded_preds, bleu_decoded_labels = postprocess_text('sacrebleu', decoded_preds, decoded_labels)
        bleu_decoded_preds, bleu_decoded_labels = list(zip(*[(p, l) for p, l in zip(bleu_decoded_preds, bleu_decoded_labels) if l[0].strip() != '']))
        try:
            bleu_result = bleu_metric.compute(predictions=bleu_decoded_preds, references=bleu_decoded_labels)
        except Exception as e:
            print(e)
            print([s for s in bleu_decoded_labels if s[0].strip() == ''])
            raise e
        result.update({"sacrebleu": bleu_result["score"]})

        # evaluate yiping bleu
        bleu_decoded_preds, bleu_decoded_labels = postprocess_text('yiping_bleu', decoded_preds, decoded_labels)
        bleu_decoded_preds, bleu_decoded_labels = list(  # 抹掉空字符串
            zip(*[(p, l) for p, l in zip(bleu_decoded_preds, bleu_decoded_labels) if l[0].strip() != '']))
        yiping_bleu_result = yiping_bleu_metric.compute(predictions=bleu_decoded_preds, references=bleu_decoded_labels,
                                                        perl_path=os.path.join(os.path.abspath(os.path.dirname(__file__)), 'metrics', 'multi-bleu-yiping.perl'))
        result.update(yiping_bleu_result)

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        fo_score = open(os.path.join(data_args.log_root, 'scores.txt'), 'a+', encoding='utf8')
        fo_score.write(f'{json.dumps(result)}\n')
        fo_score.close()
        return result

    model.config.num_beams = data_args.num_beams
    model.config.max_length = data_args.max_target_length

    callbacks = []

    # Initialize our Trainer
    training_args.remove_unused_columns = False
    trainer = MySeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        callbacks=callbacks
    )
    # Training
    if training_args.do_train:
        try:
            if data_args.fix_bart_step > 0:  # fix bart parameters at training start
                logger.info(f'fix bart parameters! These will be opened when global step > {data_args.fix_bart_step}')
                for param in model.model.parameters():
                    param.requires_grad = False
            train_result = trainer.train()  # resume_from_checkpoint=checkpoint
        except KeyboardInterrupt:
            logger.info('exit, saving model')
            trainer.save_model(output_dir=os.path.join(training_args.output_dir, f'checkpoint-{trainer.state.global_step}'))  # Saves the tokenizer too for easy upload
            trainer.state.save_to_json(os.path.join(training_args.output_dir, f'checkpoint-{trainer.state.global_step}', 'trainer_state.json'))
            exit(0)
        trainer.save_model()

        if trainer.is_world_process_zero():
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    results = {}
    if training_args.do_eval:
        if trainer.state.global_step == 0:
            trainer.state = trainer.state.load_from_json(os.path.join(model_args.model_name_or_path, "trainer_state.json"))
        logger.info(f"*** Evaluate step {trainer.state.global_step} ***")
        eval_output = trainer.evaluate()
        perplexity = math.exp(eval_output["eval_loss"])
        results["perplexity"] = perplexity
        with open(os.path.join(data_args.log_root, f'metrics.txt'), 'a+', encoding='utf8') as f:
            f.write(f'{str(results)}\n')
        if trainer.is_world_process_zero():
            logger.info("***** Eval results *****")
            for key, value in sorted(results.items()):
                logger.info(f"  {key} = {value}")

    # predict
    if training_args.do_predict:
        if trainer.state.global_step == 0 and os.path.exists(os.path.join(model_args.model_name_or_path, "trainer_state.json")):
            trainer.state = trainer.state.load_from_json(
                os.path.join(model_args.model_name_or_path, "trainer_state.json"))
        logger.info(f"*** Test ***")

        test_results = trainer.predict(
            test_dataset,
            metric_key_prefix="test",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.num_beams,
        )
        print(test_results.metrics)
        with open(os.path.join(data_args.log_root, f'metrics.txt'), 'a+', encoding='utf8') as f:
            f.write(f'{str(test_results.metrics)}\n')

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                test_results.label_ids[test_results.label_ids < 0] = tokenizer.pad_token_id
                test_label = tokenizer.batch_decode(
                    test_results.label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                test_preds = tokenizer.batch_decode(
                    test_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                test_preds = [pred.strip() for pred in test_preds]
                test_labels = [label.strip() for label in test_label]
                for pred, lab in zip(test_preds[:10], test_labels[:10]):
                    logger.info(f'{pred}\t{lab}')


if __name__ == "__main__":
    st = os.stat("./metrics/multi-bleu-yiping.perl")
    os.chmod("./metrics/multi-bleu-yiping.perl", st.st_mode | stat.S_IRGRP | stat.S_IRUSR | stat.S_IROTH |
             stat.S_IXGRP | stat.S_IXOTH | stat.S_IXUSR)
    from gpu_help import get_available_gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(get_available_gpu())
    main()
