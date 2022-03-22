import logging
import random
from typing import Dict
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def load_function(func_name: str, tokenizer: PreTrainedTokenizerBase, data_args, max_target_length, padding):
    def warp(examples: Dict):
        return globals()['_'+func_name](tokenizer, data_args, padding, max_target_length, examples)
    return warp


def _graph(tokenizer: PreTrainedTokenizerBase, data_args, padding, max_target_length, examples: Dict):
    """
    如果是json，examples就是json对应的dict。如果是纯文本，examples["text"]就是全部文本,每个item就是文本文件中的一行
    """
    # inputs = []
    # for ex_c, ex_r in zip(examples["content"], examples["retrieval"]):
    #     addi = ''
    #     for src in ['search_qa', 'search_news', 'search_point', 'search_xueqiu']:
    #         if len(ex_r[src]) > 0:
    #             addi += tokenizer.sep_token + ''.join(ex_r[src][0]['content']).replace(' ', '') if src != 'search_qa' else ex_r[src][0]['answer'].replace(' ', '')
    #     inputs.append(ex_c.replace('{', '').replace('}', '').replace(' ', '') + addi)
    if data_args.is_chinese:
        inputs = [ex.replace('{', '').replace('}', '').replace(' ', '') for ex in examples["content"]]
        targets = [ex.replace('{', '').replace('}', '').replace(' ', '') + tokenizer.eos_token for ex in examples["summary"]]
        addi_source = [
            ' '.join([
                ''.join(ex['search_xueqiu'][i]['content']).replace(' ', '') for i in range(len(ex['search_xueqiu']))
            ]) if len(ex['search_xueqiu']) > 0 else '' + tokenizer.eos_token
                       for ex in examples["retrieval"]
                    ]
    else:
        inputs = [ex.replace('{', '').replace('}', '') for ex in examples["content"]]
        targets = [ex.replace('{', '').replace('}', '') + tokenizer.eos_token for ex in examples["summary"]]
        addi_source = [ex['search_qa'][0]['answer'] if len(ex['search_qa']) > 0 else '' + tokenizer.eos_token for ex in examples["retrieval"]]

    model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True,
                             add_special_tokens=False)
    addi_source = tokenizer(addi_source, max_length=data_args.max_source_length, padding=False, truncation=True,
                            add_special_tokens=False)
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True,
                           add_special_tokens=False)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    model_inputs["addi_source"] = addi_source["input_ids"]
    model_inputs["addi_source_attention_mask"] = addi_source["attention_mask"]
    return model_inputs


def random_mask(inp: str, tokenizer) -> str:
    mask_index = random.randint(5, len(inp) - 1)
    mask_len = random.randint(1, 4)
    inp_words = list(inp)
    inp_words[mask_index] = tokenizer.mask_token
    del inp_words[mask_index + 1: mask_index + 1 + mask_len]
    inp = ''.join(inp_words)
    return inp
