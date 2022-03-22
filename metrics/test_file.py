import os
import re
from glob import glob
import stat

import nltk
from rouge_score import rouge_scorer
import numpy as np

from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

st = os.stat("../metrics/multi-bleu-yiping.perl")
os.chmod("../metrics/multi-bleu-yiping.perl", st.st_mode | stat.S_IRGRP | stat.S_IRUSR | stat.S_IROTH |
                 stat.S_IXGRP | stat.S_IXOTH | stat.S_IXUSR)

chinese = False
if chinese:
    split_char = lambda x: ' '.join(x)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
else:
    split_char = lambda x: x
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

def eval_rouge(evaluated_file, reference_file):
    fr = open(reference_file, encoding='utf8')
    fe = open(evaluated_file, encoding='utf8')
    cands = []
    refs = []
    for lr, le in zip(fr, fe):
        le = le.strip().replace('\r', '').replace('\n', '')
        lr = lr.strip().replace('\r', '').replace('\n', '')
        if le != '' and lr != '':
            if chinese:
                cands.append('\n'.join([' '.join([str(i) for i in tokenizer.encode(s, add_special_tokens=False)]) for s in nltk.sent_tokenize(le.strip())]))
                refs.append('\n'.join([' '.join([str(i) for i in tokenizer.encode(s, add_special_tokens=False)]) for s in nltk.sent_tokenize(lr.strip())]))
            else:
                # cands.append('\n'.join(
                #     [' '.join([str(i) for i in tokenizer.encode(s, add_special_tokens=False)]) for s in
                #      nltk.sent_tokenize(le.strip())]))
                # refs.append('\n'.join(
                #     [' '.join([str(i) for i in tokenizer.encode(s, add_special_tokens=False)]) for s in
                    #  nltk.sent_tokenize(lr.strip())]))
                
                # refs.append(' '.join([str(i) for i in tokenizer.encode(lr, add_special_tokens=False)]))
                # cands.append(' '.join([str(i) for i in tokenizer.encode(le, add_special_tokens=False)]))
                cands.append('\n'.join([s for s in nltk.sent_tokenize(le.strip())]))
                refs.append('\n'.join([s for s in nltk.sent_tokenize(lr.strip())]))
    assert len(cands) == len(refs)
    aggregator = rouge_scorer.scoring.BootstrapAggregator()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum', 'rougeL'], use_stemmer=False)
    for r_sent, c_sent in zip(refs, cands):
        aggregator.add_scores(scorer.score(r_sent, c_sent))
    return {f'c-{k}': v.high.fmeasure*100 for k, v in aggregator.aggregate().items()}

def sys_bleu_perl_file(evaluated_file, reference_file):
    fi = open(f'{evaluated_file}', encoding='utf8')
    fo = open(f'{evaluated_file}.new', 'w', encoding='utf8')
    for line in fi:
        # fo.write(f'{line}\n')
        fo.write(f'{split_char(line)}\n')
    fi.close()
    fo.close()
    fi = open(f'{reference_file}', encoding='utf8')
    fo = open(f'{reference_file}.new', 'w', encoding='utf8')
    for line in fi:
        # fo.write(f'{line}\n')
        fo.write(f'{split_char(line)}\n')
    fi.close()
    fo.close()
    command = '../metrics/multi-bleu-yiping.perl' + ' %s.new < %s.new' % (reference_file, evaluated_file)
    result = os.popen(command).readline().strip()
    BLEU_value_pattern = re.compile(
        'BLEU = (\d+.\d+?), (\d+.\d+?), (\d+.\d+?), (\d+.\d+?), (\d+.\d+?) \(BP=(\d+.\d+?), ratio=(\d+.\d+?), hyp_len=(\d+.\d+?), ref_len=(\d+.\d+?)\)')
    value_entry = BLEU_value_pattern.findall(result)[0]
    names = ['blue', 'blue1', 'blue2', 'blue3', 'blue4', 'bp', 'ratio', 'hyp_len', 'ref_len']
    value_dict = {n: float(v) for n, v in zip(names, value_entry)}
    os.remove(f'{reference_file}.new')
    os.remove(f'{evaluated_file}.new')
    return value_dict

def eval_meteor(evaluated_file, reference_file):
    from nltk.translate import meteor_score
    alpha = 0.9
    beta = 3
    gamma = 0.5
    fr = open(reference_file, encoding='utf8')
    fe = open(evaluated_file, encoding='utf8')
    cands = []
    refs = []
    for lr, le in zip(fr, fe):
        le = le.strip().replace('\r', '').replace('\n', '')
        lr = lr.strip().replace('\r', '').replace('\n', '')
        if le != '' and lr != '':
            cands.append(split_char(le.strip()))
            refs.append(split_char(lr.strip()))
    assert len(cands) == len(refs)
    scores = [
        meteor_score.single_meteor_score(ref, pred, alpha=alpha, beta=beta, gamma=gamma)
        for ref, pred in zip(refs, cands)
    ]

    return {"meteor": np.mean(scores)}


if __name__ == '__main__':
    bb = './msm-plus-decode/'
    dd = eval_rouge(f'{bb}decoded.txt', f'{bb}reference.txt')
    dd2 = eval_meteor(f'{bb}decoded.txt', f'{bb}reference.txt')
    dd3 = sys_bleu_perl_file(f'{bb}decoded.txt', f'{bb}reference.txt')
    print(dd)
    print(dd2)
    print(dd3)

    # for pp in glob('msm-decode-baselines/*'):
    #     dec = f'{pp}/decoded.txt'
    #     ref = f'{pp}/reference.txt'
    #     dd = eval_rouge(dec, ref)
    #     dd2 = eval_meteor(dec, ref)
    #     dd3 = sys_bleu_perl_file(dec, ref)
    #     dd.update(dd2)
    #     dd.update(dd3)
    #     print(pp, dd)
