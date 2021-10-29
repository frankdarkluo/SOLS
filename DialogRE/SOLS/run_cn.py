from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
from operator import pos, sub
import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import spacy
from collections import Counter
import jieba
import constant
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from modeling import MainSequenceClassification
import json
import warnings
import pickle

warnings.filterwarnings('ignore')

n_class = 1
reverse_order = False
sa_step = False

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


# torch.cuda.set_device(0)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, text_c=None, x_type=None, y_type=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label
        self.x_type = x_type
        self.y_type = y_type


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, subj_positions=None, obj_positions=None,
                 pos_ids=None, x_type=None, y_type=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.subj_positions = subj_positions
        self.obj_positions = obj_positions
        self.pos_ids = pos_ids
        self.x_type = x_type
        self.y_type = y_type


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class gloveLSTMProcessor(DataProcessor):
    def __init__(self):
        random.seed(42)
        self.D = [[], [], []]
        for sid in range(3):
            with open("datacn/" + ["train.json", "dev.json", "test.json"][sid], "r", encoding="utf8") as f:
                data = json.load(f)
            if sid == 0:
                random.shuffle(data)
            for i in range(len(data)):
                for j in range(len(data[i][1])):
                    rid = []
                    for k in range(36):
                        if k + 1 in data[i][1][j]["rid"]:
                            rid += [1]
                        else:
                            rid += [0]
                    d = [' '.join(data[i][0]).lower(),
                         data[i][1][j]["x"].lower(),
                         data[i][1][j]["y"].lower(),
                         rid,
                         data[i][1][j]["x_type"],
                         data[i][1][j]["y_type"]]
                    self.D[sid] += [d]
        logger.info(str(len(self.D[0])) + "," + str(len(self.D[1])) + "," + str(len(self.D[2])))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.D[0], "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.D[2], "test")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.D[1], "dev")

    def get_labels(self):
        """See base class."""
        return [str(x) for x in range(2)]

    def _create_examples(self, data, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, d) in enumerate(data):
            guid = "%s-%s" % (set_type, i)
            examples.append(
                InputExample(guid=guid, text_a=data[i][0], text_b=data[i][1], label=data[i][3], text_c=data[i][2],
                             x_type=data[i][4], y_type=data[i][5]))

        return examples


class gloveLSTMf1cProcessor(DataProcessor):
    def __init__(self):
        random.seed(42)
        self.D = [[], [], []]
        for sid in range(1, 3):
            with open("datacn/" + ["dev.json", "test.json"][sid - 1], "r", encoding="utf8") as f:
                data = json.load(f)
            for i in range(len(data)):
                for j in range(len(data[i][1])):
                    rid = []
                    for k in range(36):
                        if k + 1 in data[i][1][j]["rid"]:
                            rid += [1]
                        else:
                            rid += [0]
                    for l in range(1, len(data[i][0]) + 1):
                        d = [' '.join(data[i][0][:l]).lower(),
                             data[i][1][j]["x"].lower(),
                             data[i][1][j]["y"].lower(),
                             rid,
                             data[i][1][j]["x_type"],
                             data[i][1][j]["y_type"]]
                        self.D[sid] += [d]
        logger.info(str(len(self.D[0])) + "," + str(len(self.D[1])) + "," + str(len(self.D[2])))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.D[0], "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.D[2], "test")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.D[1], "dev")

    def get_labels(self):
        """See base class."""
        return [str(x) for x in range(2)]

    def _create_examples(self, data, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, d) in enumerate(data):
            guid = "%s-%s" % (set_type, i)
            examples.append(
                InputExample(guid=guid, text_a=data[i][0], text_b=data[i][1], label=data[i][3], text_c=data[i][2],
                             x_type=data[i][4], y_type=data[i][5]))

        return examples


def tokenize(text, tokenizer):
    D = ['[unused1]', '[unused2]']
    text_tokens = []
    textraw = [text]
    for delimiter in D:
        ntextraw = []
        for i in range(len(textraw)):
            t = textraw[i].split(delimiter)
            for j in range(len(t)):
                ntextraw += [t[j]]
                if j != len(t) - 1:
                    ntextraw += [delimiter]
        textraw = ntextraw
    text = []
    for t in textraw:
        if t in ['[unused1]', '[unused2]']:
            text += [t]
        else:
            tokens = tokenizer.tokenize(t)
            for tok in tokens:
                text += [tok]
    return text


class Vocab(object):
    def __init__(self, init_wordlist, word_counter):
        self.word2id = {w: i for i, w in enumerate(init_wordlist)}
        self.id2word = {i: w for i, w in enumerate(init_wordlist)}
        self.n_words = len(init_wordlist)
        self.word_counter = word_counter

    def map(self, token_list):
        """
        Map a list of tokens to their ids.
        """
        return [self.word2id[w] if w in self.word2id else constant.UNK_ID for w in token_list]

    def unmap(self, idx_list):
        """
        Unmap ids back to tokens.
        """
        return [self.id2word[idx] for idx in idx_list]


def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids

# tokenize Chinese words
def tok_list(word):
    tok = []
    for i in word:
        if is_Chinese(i):
            a = [a for a in i]
            tok += a
        elif not is_Chinese(i) and i != ' ':
            tok.append(i)
    return tok


def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False


def convert_examples_to_feats_lstm(examples, max_seq_length, glove_vocab, feat_file, language):
    """Loads a data file into a list of `InputBatch`s in glove+lstm manner"""

    print("#examples", len(examples))
    if os.path.exists(feat_file):
        with open(feat_file, 'rb') as f:
            features = pickle.load(f)
        return features
    else:
        features = [[]]
        if language == 'cn':
            nlp = spacy.load("zh_core_web_md")
        for (ex_index, example) in enumerate(examples):
            abandon = False
            if language == 'cn':
                dialog = tok_list([token for token in jieba.cut(example.text_a, cut_all=False)])
                utter = nlp(' '.join(dialog))
                dialog_tokens = [token.text for token in utter]
                a_tokens = tok_list([token for token in jieba.cut(example.text_b, cut_all=False)])
                b_tokens = tok_list([token for token in jieba.cut(example.text_c, cut_all=False)])
                dialog_pos = [token.tag_ for token in utter]
                assert len(dialog_tokens) == len(dialog_pos)

            x_type = [example.x_type]
            y_type = [example.y_type]

            truncate(dialog_tokens, max_seq_length)
            truncate(dialog_pos, max_seq_length)

            a_start = []
            a_end = []

            for i in range(len(dialog_tokens) - len(a_tokens)):
                match = 0
                for j in range(len(a_tokens)):
                    if dialog_tokens[i + j] == a_tokens[j]:
                        match += 1
                if match == len(a_tokens):
                    a_start.append(i)
                    a_end.append(i + match - 1)

            if len(a_start) == 0:
                a_start.append(0)
                a_end.append(len(a_tokens)-1)

            b_start = []
            b_end = []

            for i in range(len(dialog_tokens) - len(b_tokens)):
                match = 0
                for j in range(len(b_tokens)):
                    if dialog_tokens[i + j] == b_tokens[j]:
                        match += 1
                if match == len(b_tokens):
                    b_start.append(i)
                    b_end.append(i + match - 1)

            if len(b_start) == 0:
                b_start.append(0)
                b_end.append(len(b_tokens)-1)

            # l = len(dialog_tokens)
            subj_positions = [1] * max_seq_length
            obj_positions = [1] * max_seq_length

            for s, e in zip(a_start, a_end):
                subj_positions[s] = 0
                subj_positions[e] = 0

            for s, e in zip(b_start, b_end):
                obj_positions[s] = 0
                obj_positions[e] = 0

            # convert tokens to index
            input_ids = glove_vocab.map(dialog_tokens)
            # convert pos to index
            pos_ids = map_to_ids(dialog_pos, constant.POS_TO_ID)
            # convert ner to index
            x_type = map_to_ids(x_type, constant.NER_TO_ID)
            y_type = map_to_ids(y_type, constant.NER_TO_ID)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)  # actually not used

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                pos_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(pos_ids) == max_seq_length

            label_id = example.label

            if ex_index < 2:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("tokens: %s" % " ".join([str(token) for token in dialog_tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

            if not abandon:
                features[-1].append(
                    InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_id,
                        subj_positions=subj_positions,
                        obj_positions=obj_positions,
                        pos_ids=pos_ids,
                        x_type=x_type,
                        y_type=y_type))
                if len(features[-1]) == n_class:
                    features.append([])

        if len(features[-1]) == 0:
            features = features[:-1]
        print('#features', len(features))
        with open(feat_file, 'wb') as f:
            pickle.dump(features, f)

        return features


def truncate(tokens, max_length):
    while True:
        total_length = len(tokens)
        if total_length <= max_length:
            break
        else:
            tokens.pop()


def _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence tuple in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) >= len(tokens_b) and len(tokens_a) >= len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) >= len(tokens_a) and len(tokens_b) >= len(tokens_c):
            tokens_b.pop()
        else:
            tokens_c.pop()


def load_all_tokens(args):
    all_set_tokens = []
    language = args.language
    if os.path.exists(args.token_pkl):
        print("LOADING dialogre dataset")
        with open(args.token_pkl, "rb") as f:
            all_set_tokens = pickle.load(f)
        return all_set_tokens
    else:
        if language == 'cn':
            nlp = spacy.load("zh_core_web_md")
            for sid in range(3):
                with open("datacn/" + ["train.json", "dev.json", "test.json"][sid], "r", encoding="utf8") as f:
                    data = json.load(f)
                    for i in range(len(data)):
                        dialog = ' '.join(data[i][0])
                        dialog_tokens = tok_list([token for token in jieba.cut(dialog, cut_all=False)])
                        all_set_tokens += dialog_tokens

        with open(args.token_pkl, 'wb') as f:
            pickle.dump(all_set_tokens, f)

        return all_set_tokens


def load_glove_vocab(glove_file, wv_dim):
    vocab = set()

    with open(glove_file, encoding='utf8') as f:
        for line in f:
            elems = line.split()
            token = ''.join(elems[0:-wv_dim])
            vocab.add(token)

    return vocab


def build_vocab(tokens, glove_vocab, args, min_freq=0):
    """ build vocab from tokens and glove words. """
    counter = Counter(t for t in tokens)
    # if min_freq > 0, use min_freq, otherwise keep all glove words
    if args.min_freq > 0:
        v = sorted([t for t in counter if counter.get(t) >= min_freq], key=counter.get, reverse=True)
    else:
        v = sorted([t for t in counter if t in glove_vocab], key=counter.get, reverse=True)
    # add special tokens and entity mask tokens
    v = constant.VOCAB_PREFIX + v
    print("vocab built with {}/{} words.".format(len(v), len(counter)))
    return v, counter


def build_embedding(wv_file, vocab, wv_dim, args):
    vocab_size = len(vocab)
    emb = np.random.randn(vocab_size, args.embed_dim) * 0.01
    emb[constant.PAD_ID] = 0  # <pad> should be all 0

    w2id = {w: i for i, w in enumerate(vocab)}
    with open(wv_file, encoding="utf8") as f:
        for line in f:
            elems = line.split()
            token = ''.join(elems[0:-wv_dim])
            if token in w2id:
                emb[w2id[token]] = [float(v) for v in elems[-wv_dim:]]

    return emb


def accuracy(out, labels, dataset):
    out = out.reshape(-1)
    out = 1 / (1 + np.exp(-out))
    if dataset == "dialogre":
        res = np.sum((out > 0.5) == (labels > 0.5)) / 36
    return res


def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)


def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if test_nan and torch.isnan(param_model.grad).sum() > 0:
            is_nan = True
        if param_opti.grad is None:
            param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
        param_opti.grad.data.copy_(param_model.grad.data)
    return is_nan


def f1_eval(logits, features, dataset):
    if dataset == "dialogre":
        NUM = 36

    def getpred(result, T1=0.5, T2=0.4):
        ret = []
        for i in range(len(result)):
            r = []
            maxl, maxj = -1, -1
            for j in range(len(result[i])):
                if result[i][j] > T1:
                    r += [j]
                if result[i][j] > maxl:
                    maxl = result[i][j]
                    maxj = j
            if len(r) == 0:
                if maxl <= T2:
                    r = [NUM]
                else:
                    r += [maxj]
            ret += [r]
        return ret

    def geteval(devp, data):
        correct_sys, all_sys = 0, 0
        correct_gt = 0

        for i in range(len(data)):
            for id in data[i]:
                if id != NUM:
                    correct_gt += 1
                    if id in devp[i]:
                        correct_sys += 1

            for id in devp[i]:
                if id != NUM:
                    all_sys += 1

        precision = 1 if all_sys == 0 else correct_sys / all_sys
        recall = 0 if correct_gt == 0 else correct_sys / correct_gt
        f_1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        return f_1

    logits = np.asarray(logits)
    logits = list(1 / (1 + np.exp(-logits)))

    labels = []
    for f in features:
        label = []
        assert (len(f[0].label_id) == NUM)
        for i in range(NUM):
            if f[0].label_id[i] == 1:
                label += [i]
        if len(label) == 0:
            label = [NUM]
        labels += [label]
    assert (len(labels) == len(logits))

    bestT2 = bestf_1 = 0
    for T2 in range(51):
        devp = getpred(logits, T2=T2 / 100.)
        f_1 = geteval(devp, labels)
        if f_1 > bestf_1:
            bestf_1 = f_1
            bestT2 = T2 / 100.

    return bestf_1, bestT2


def read_lstm_features(features):
    input_ids = []
    input_mask = []
    segment_ids = []
    label_id = []

    subj_positions = []
    obj_positions = []
    pos_ids = []
    x_type = []
    y_type = []

    for f in features:
        input_ids.append([])
        input_mask.append([])
        segment_ids.append([])

        subj_positions.append([])
        obj_positions.append([])
        pos_ids.append([])
        x_type.append([])
        y_type.append([])
        for i in range(1):
            input_ids[-1].append(f[i].input_ids)
            input_mask[-1].append(f[i].input_mask)
            segment_ids[-1].append(f[i].segment_ids)

            subj_positions[-1].append(f[i].subj_positions)
            obj_positions[-1].append(f[i].obj_positions)
            pos_ids[-1].append(f[i].pos_ids)
            x_type[-1].append(f[i].x_type)
            y_type[-1].append(f[i].y_type)
        label_id.append([f[0].label_id])

    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(input_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(segment_ids, dtype=torch.long)
    all_label_ids = torch.tensor(label_id, dtype=torch.float)

    all_subj_positions = torch.tensor(subj_positions, dtype=torch.long)
    all_obj_positions = torch.tensor(obj_positions, dtype=torch.long)
    all_pos_ids = torch.tensor(pos_ids, dtype=torch.long)
    all_x_type = torch.tensor(x_type, dtype=torch.long)
    all_y_type = torch.tensor(y_type, dtype=torch.long)

    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_subj_positions,
                         all_obj_positions, all_pos_ids, all_x_type, all_y_type)

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default='.',
                        type=str,
                        # required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task_name",
                        default='lstm',
                        type=str,
                        # required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default='lstmcn_f1',
                        type=str,
                        # required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size",
                        default=4,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=4,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=1e-4,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=20.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--save_checkpoints_steps",
                        default=1000,
                        type=int,
                        help="How often to save the model checkpoint.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help="random seed for initialization")

    parser.add_argument('--pooling_ratio', type=float, default=0.3,
                        help='pooling ratio')
    parser.add_argument('--pool_dropout_ratio', type=float, default=0.5,
                        help='dropout ratio')
    parser.add_argument('--rnn_hidden_size',
                        type=int,
                        default=384,
                        help="Hidden_size for rnn")
    parser.add_argument('--graph_hidden_size',
                        type=int,
                        default=300,
                        help="Hidden_size for graph")
    parser.add_argument('--input_dropout',
                        type=float, default=0.5,
                        help='Dropout rate for word representation.')
    parser.add_argument('--num_graph_layers',
                        type=int,
                        default=2,
                        help="Number of blocks for graph")
    parser.add_argument('--heads', type=int, default=3, help='Num of heads in multi-head attention.')
    parser.add_argument('--sublayer_first', type=int, default=2, help='Num of the first sublayers in dcgcn block.')
    parser.add_argument('--sublayer_second', type=int, default=4, help='Num of the second sublayers in dcgcn block.')
    parser.add_argument('--gcn_dropout', type=float, default=0.5, help='AGGCN layer dropout rate.')
    parser.add_argument('--lamada', type=float, default=0.000001, help='Weights for DTW Loss.')
    parser.add_argument('--max_offset', type=int, default=4, help='Length of max_offset.')

    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument("--resume",
                        default=False,
                        action='store_true',
                        help="Whether to resume the training.")
    parser.add_argument("--f1eval",
                        default=True,
                        action='store_true',
                        help="Whether to use f1 for dev evaluation during training.")

    parser.add_argument('--glove_f', type=str, default='datacn/sgns.wiki.bigram-char.txt')
    parser.add_argument('--embed_dim', type=int, default=300, help='Word embedding dimension.')
    parser.add_argument('--encoder_type', type=str, default='LSTM', help='LSTM')
    parser.add_argument('--embed_f', type=str, default='datacn/embeddings.npy')
    parser.add_argument('--min_freq', type=int, default=1, help='Minimal word frequency for builiding vocab')
    parser.add_argument('--tune_topk', type=int, default=1e10, help='Only finetune top N word embeddings.')
    parser.add_argument('--lstm_layers', type=int, default=1, help='Number of lstm layers')
    parser.add_argument('--lstm_dropout', type=int, default=0.2, help='dropout rate of lstm')
    parser.add_argument("--lstm_only", default=False, action='store_true', help="Whether to only use BiLSTM")
    parser.add_argument('--weight_decay', type=float, default=0.2, help='dropout rate of lstm')
    parser.add_argument('--token_pkl', type=str, default='datacn/token_pkl', help='pickle file for all tokens')
    parser.add_argument('--vocab_pkl', type=str, default='datacn/vocab_pkl', help='pickle file for vocab')
    parser.add_argument('--vocab_ct_pkl', type=str, default='datacn/vocab_ct_pkl', help='pickle file for vocab counter')
    parser.add_argument('--train_feat_pkl', type=str, default='datacn/train_feat_pkl', help='pickle file for train')
    parser.add_argument('--eval_feat_pkl', type=str, default='datacn/eval_feat_pkl', help='pickle file for eval')
    parser.add_argument('--test_feat_pkl', type=str, default='datacn/test_feat_pkl', help='pickle file for test')
    parser.add_argument('--train_feat_c_pkl', type=str, default='datacn/train_feat_c_pkl', help='pickle file for train')
    parser.add_argument('--eval_feat_c_pkl', type=str, default='datacn/eval_feat_c_pkl', help='pickle file for eval')
    parser.add_argument('--test_feat_c_pkl', type=str, default='datacn/test_feat_c_pkl', help='pickle file for test')

    parser.add_argument('--num_layer', type=int, default=1, help='layer number for latent structure')
    parser.add_argument('--first_layer', type=int, default=2, help='first layer')
    parser.add_argument('--second_layer', type=int, default=3, help='second layer')
    parser.add_argument('--latent_dropout', type=float, default=0.2, help="dropout for latent structure")
    parser.add_argument('--diff_mlp_hidden', type=int, default=128, help="MLP inter-mediate hidden size")
    parser.add_argument('--diff_position', type=int, default=500, help="max position number")
    parser.add_argument('--latent_heads', type=int, default=4, help="heads for multihead attention")

    parser.add_argument('--dropout_rate', type=float, default=0.3, help="dropout rate for classifier")
    parser.add_argument('--rm_stopwords', type=bool, default=False, help='Remove stopwords in global word Node')

    parser.add_argument('--latent_type', type=str, default='sols',
                        help="['None','sols']")
    parser.add_argument('--extract_node_id', type=bool, default=True, help='extract node id')

    parser.add_argument('--l0_reg', type=bool, default=False, help='l0 regualrization')
    parser.add_argument('--speaker_reg', type=bool, default=True, help='speaker-related regualrization')
    parser.add_argument('--lasso_reg', type=bool, default=True, help='sparsity regualrization')
    parser.add_argument('--alpha', type=float, default=0.01, help="weight for l0")
    parser.add_argument('--beta', type=float, default=0.01, help="weight for speaker reg")
    parser.add_argument('--gamma', type=float, default=0.01, help="weight for lasso reg")
    parser.add_argument('--num_latent_graphs', type=int, default=1, help='1, or 2')
    
    # language
    parser.add_argument('--language', type=str, default='cn', help="en | cn ")
    parser.add_argument('--dataset', type=str, default='dialogre', help="dialogre")

    args = parser.parse_args()

    processors = {
        "lstm": gloveLSTMProcessor,
        "lstmf1c": gloveLSTMf1cProcessor
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False  # (see https://github.com/pytorch/pytorch/pull/13496)

    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and 'model.pt' in os.listdir(args.output_dir):
        if args.do_train and not args.resume:
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    else:
        os.makedirs(args.output_dir, exist_ok=True)
    
    if args.do_train: 
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "train.log"), 'w')) 
    else: 
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w')) 
    
    logger.info(args) 

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    # get and process all data
    processor = processors[task_name]()
    label_list = processor.get_labels()

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_steps = int(
            len(
                train_examples) / n_class / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    vocab = None
    # components needed for lstm
    # load all tokens in train, dev, and test dataset
    logger.info("loading all tokens in dataset ...")
    all_tokens = load_all_tokens(args)
    # load glove vocab and embedding
    logger.info("loading sgns vocab ...")
    if os.path.exists(args.vocab_pkl):
        print("LOADING vocab dataset")
        with open(args.vocab_pkl, "rb") as f:
            v = pickle.load(f)
        with open(args.vocab_ct_pkl, 'rb') as fc:
            v_counter = pickle.load(fc)
    else:
        glove_vocab = load_glove_vocab(args.glove_f, args.embed_dim)
        logger.info("{} words loaded from glove.".format(len(glove_vocab)))
        # build vocab
        logger.info("building vocab")
        v, v_counter = build_vocab(all_tokens, glove_vocab, args=args)
        with open(args.vocab_pkl, 'wb') as fv:
            pickle.dump(v, fv)
        with open(args.vocab_ct_pkl, 'wb') as fc:
            pickle.dump(v_counter, fc)

    vocab = Vocab(v, v_counter)
    # build embedding and write to a file
    logger.info("building sgns embeddings ...")
    if os.path.exists(args.vocab_pkl) == True:
        embedding = build_embedding(args.glove_f, v, args.embed_dim, args)
        logger.info("embedding size: {} x {}".format(*embedding.shape))
        logger.info("dumping embedding file ...")
        np.save(args.embed_f, embedding)

    model = MainSequenceClassification( 1, vocab, args)

    if args.fp16:
        model.half()
    model.to(device)

    if args.fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                           for n, param in model.named_parameters()]
    elif args.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                           for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if n not in no_decay], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if n in no_decay], 'weight_decay_rate': 0.0}
    ]

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    global_step = 0

    if args.resume:
        model.load_state_dict(torch.load(os.path.join(args.output_dir, "model.pt")))

    if args.do_eval:
        eval_examples = processor.get_dev_examples(args.data_dir)
        if args.task_name== 'lstm':
            eval_features = convert_examples_to_feats_lstm(eval_examples, args.max_seq_length, vocab,
                                                            args.eval_feat_pkl, args.language)
        elif args.task_name=='lstmf1c':
            eval_features = convert_examples_to_feats_lstm(eval_examples, args.max_seq_length, vocab,
                                                            args.eval_feat_c_pkl, args.language)

        eval_data = read_lstm_features(eval_features)

        if args.local_rank == -1:
            eval_sampler = SequentialSampler(eval_data)
        else:
            eval_sampler = DistributedSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    if args.do_train:
        best_metric = 0
        if args.task_name=='lstm':
            train_features = convert_examples_to_feats_lstm(train_examples, args.max_seq_length, vocab,
                                                            args.train_feat_pkl, args.language)
        elif args.task_name=="lstmf1c":
            train_features = convert_examples_to_feats_lstm(train_examples, args.max_seq_length, vocab,
                                                            args.train_feat_c_pkl, args.language)

        train_data = read_lstm_features(train_features)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(
                   tqdm(train_dataloader, desc="Iteration")):
            # for step, batch in enumerate(train_dataloader):  # (tqdm(train_dataloader, desc="Iteration")):

                batch = tuple(t.to(device) for t in batch)

                input_ids, input_mask, segment_ids, label_ids, subj_position, obj_position, pos_ids, x_type, y_type = batch
                loss, _ = model(input_ids, segment_ids, input_mask, subj_position, obj_position, pos_ids, x_type,
                                y_type, label_ids, 1)

                if n_gpu > 1:
                    loss = loss.mean()
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16 or args.optimize_on_cpu:
                        if args.fp16 and args.loss_scale != 1.0:
                            # scale down gradients for fp16 training
                            for param in model.parameters():
                                param.grad.data = param.grad.data / args.loss_scale
                        is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                        if is_nan:
                            logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                            args.loss_scale = args.loss_scale / 2
                            model.zero_grad()
                            continue
                        optimizer.step()
                        copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                    else:
                        optimizer.step()
                    model.zero_grad()
                    global_step += 1

            model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            logits_all = []

            for input_ids, input_mask, segment_ids, label_ids, subj_position, obj_position, pos_ids, x_type, y_type in eval_dataloader:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)
                subj_position = subj_position.to(device)
                obj_position = obj_position.to(device)
                pos_ids = pos_ids.to(device)
                x_type = x_type.to(device)
                y_type = y_type.to(device)

                with torch.no_grad():
                    tmp_eval_loss, logits = model(input_ids, segment_ids, input_mask, subj_position, obj_position,
                                                    pos_ids, x_type, y_type, label_ids, 1)

                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                for i in range(len(logits)):
                    logits_all += [logits[i]]

                tmp_eval_accuracy = accuracy(logits, label_ids.reshape(-1), args.dataset)

                eval_loss += tmp_eval_loss.mean().item()
                eval_accuracy += tmp_eval_accuracy

                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1


            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples

            if args.do_train:
                result = {'eval_loss': eval_loss,
                          'global_step': global_step,
                          'loss': tr_loss / nb_tr_steps}
            else:
                result = {'eval_loss': eval_loss}

            if args.f1eval:
                eval_f1, eval_T2 = f1_eval(logits_all, eval_features, args.dataset)
                result["f1"] = eval_f1
                result["T2"] = eval_T2
                with open(os.path.join(args.output_dir, 'dev_result.json'), 'w', encoding='utf8') as of:
                    json.dump(result, of, indent=2, ensure_ascii=False)

            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))

            if args.f1eval:
                if eval_f1 >= best_metric:
                    torch.save(model.state_dict(), os.path.join(args.output_dir, "model_best.pt"))
                    best_metric = eval_f1
            else:
                if eval_accuracy >= best_metric:
                    torch.save(model.state_dict(), os.path.join(args.output_dir, "model_best.pt"))
                    best_metric = eval_accuracy

        model.load_state_dict(torch.load(os.path.join(args.output_dir, "model_best.pt")))
        torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))

    model.load_state_dict(torch.load(os.path.join(args.output_dir, "model.pt")))

    if args.do_eval:
        logger.info("***** Running dev evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        model.eval()
        eval_loss = 0
        nb_eval_steps, nb_eval_examples = 0, 0
        logits_all = []
        for input_ids, input_mask, segment_ids, label_ids, subj_position, obj_position, pos_ids, x_type, y_type in eval_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            pos_ids = pos_ids.to(device)
            x_type = x_type.to(device)
            y_type = y_type.to(device)

            with torch.no_grad():
                tmp_eval_loss, logits = model(input_ids, segment_ids, input_mask, subj_position, obj_position,
                                                pos_ids, x_type, y_type, label_ids, 1)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            for i in range(len(logits)):
                logits_all += [logits[i]]

            eval_loss += tmp_eval_loss.mean().item()

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps


        if args.do_train:
            result = {'eval_loss': eval_loss,
                      'global_step': global_step,
                      'loss': tr_loss / nb_tr_steps}
        else:
            result = {'eval_loss': eval_loss}

        if args.f1eval:
            eval_f1, eval_T2 = f1_eval(logits_all, eval_features, args.dataset)
            result["f1"] = eval_f1
            result["T2"] = eval_T2

        output_eval_file = os.path.join(args.output_dir, "eval_results_dev.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
        output_eval_file = os.path.join(args.output_dir, "logits_dev.txt")
        with open(output_eval_file, "w") as f:
            for i in range(len(logits_all)):
                for j in range(len(logits_all[i])):
                    f.write(str(logits_all[i][j]))
                    if j == len(logits_all[i]) - 1:
                        f.write("\n")
                    else:
                        f.write(" ")

        eval_examples = processor.get_test_examples(args.data_dir)
        if args.task_name=="lstm":
            eval_features = convert_examples_to_feats_lstm(eval_examples, args.max_seq_length, vocab,
                                                            args.test_feat_pkl, args.language)
        elif args.task_name=="lstmf1c":
            eval_features = convert_examples_to_feats_lstm(eval_examples, args.max_seq_length, vocab,
                                                            args.test_feat_c_pkl, args.language)

        eval_data = read_lstm_features(eval_features)

        logger.info("***** Running test evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        if args.local_rank == -1:
            eval_sampler = SequentialSampler(eval_data)
        else:
            eval_sampler = DistributedSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss = 0
        nb_eval_steps, nb_eval_examples = 0, 0
        logits_all = []

        for input_ids, input_mask, segment_ids, label_ids, subj_position, obj_position, pos_ids, x_type, y_type in eval_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            subj_position = subj_position.to(device)
            obj_position = obj_position.to(device)
            pos_ids = pos_ids.to(device)
            x_type = x_type.to(device)
            y_type = y_type.to(device)

            with torch.no_grad():
                tmp_eval_loss, logits = model(input_ids, segment_ids, input_mask, subj_position, obj_position,
                                                pos_ids,
                                                x_type, y_type, label_ids, 1)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            for i in range(len(logits)):
                logits_all += [logits[i]]

            eval_loss += tmp_eval_loss.mean().item()

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps

        if args.do_train:
            result = {'eval_loss': eval_loss,
                      'global_step': global_step,
                      'loss': tr_loss / nb_tr_steps}
        else:
            result = {'eval_loss': eval_loss}

        if args.f1eval:
            test_f1, test_T2 = f1_eval(logits_all, eval_features, args.dataset)
            result["f1"] = test_f1
            result["T2"] = test_T2

        output_eval_file = os.path.join(args.output_dir, "eval_results_test.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
        output_eval_file = os.path.join(args.output_dir, "logits_test.txt")
        with open(output_eval_file, "w") as f:
            for i in range(len(logits_all)):
                for j in range(len(logits_all[i])):
                    f.write(str(logits_all[i][j]))
                    if j == len(logits_all[i]) - 1:
                        f.write("\n")
                    else:
                        f.write(" ")

if __name__ == "__main__":
    main()
