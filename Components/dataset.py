from collections import OrderedDict
import torch
import numpy as np

import pickle

from torch.autograd import Variable
from .nn_utils import cached_property
from . import nn_utils


class SQLDataset(object):
    def __init__(self, examples):
        self.examples = examples

    @property
    def all_source(self):
        return [e.src_sent for e in self.examples]

    @property
    def all_targets(self):
        return [e.tgt_code for e in self.examples]

    @staticmethod
    def from_bin_file(file_path):
        examples = pickle.load(open(file_path, 'rb'), encoding='utf-8')
        return SQLDataset(examples)

    def batch_iter(self, batch_size, shuffle=False):
        index_arr = np.arange(len(self.examples))
        if shuffle:
            np.random.shuffle(index_arr)

        batch_num = int(np.ceil(len(self.examples) / float(batch_size)))
        for batch_id in range(batch_num):
            batch_ids = index_arr[batch_size * batch_id: batch_size * (batch_id + 1)]
            batch_examples = [self.examples[i] for i in batch_ids]
            batch_examples.sort(key=lambda e: -len(e.src_sent))

            yield batch_examples

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)


class SQLExample(object):
    def __init__(self, src_sent, tgt_grammar, tgt_code, tgt_dfa, idx=0, meta=None):
        self.src_sent = src_sent  # SQL query in natural language
        self.tgt_code = tgt_code  # Target SQL code
        self.tgt_dfa = tgt_dfa    # DFA specific to SQL parsing
        self.tgt_grammar = tgt_grammar  # SQL grammar rules

        self.idx = idx
        self.meta = meta


class SQLBatch(object):
    def __init__(self, examples, grammar, vocab, copy=True, cuda=False):
        self.examples = examples
        self.max_code_num = max(len(e.tgt_code) for e in self.examples)

        self.src_sents = [e.src_sent for e in self.examples]
        self.src_sents_len = [len(e.src_sent) for e in self.examples]

        self.grammar = grammar
        self.vocab = vocab
        self.copy = copy
        self.cuda = cuda

        self.init_index_tensors()

    def __len__(self):
        return len(self.examples)

    def init_index_tensors(self):
        self.apply_rule_idx_matrix = []
        self.apply_rule_mask = []
        self.primitive_idx_matrix = []
        self.gen_token_mask = []
        self.primitive_copy_mask = []
        self.primitive_copy_token_idx_mask = np.zeros((self.max_code_num, len(self), max(self.src_sents_len)), dtype='float32')

        for t in range(self.max_code_num):
            app_rule_idx_row = []
            app_rule_mask_row = []
            token_row = []
            gen_token_mask_row = []
            copy_mask_row = []

            for e_id, e in enumerate(self.examples):
                app_rule_idx = app_rule_mask = token_idx = gen_token_mask = copy_mask = 0
                if t < len(e.tgt_code):
                    app_rule_idx = self.grammar.nonterminal2id[e.tgt_grammar[t]]
                    app_rule_mask = 1
                    
                    src_sent = self.src_sents[e_id]
                    token = e.tgt_code[t][0]
                    token_idx = self.vocab.code[e.tgt_code[t]]

                    token_can_copy = False

                    if self.copy and token in src_sent:
                        token_pos_list = [idx for idx, _token in enumerate(src_sent) if _token == token]
                        self.primitive_copy_token_idx_mask[t, e_id, token_pos_list] = 1.
                        copy_mask = 1
                        token_can_copy = True

                    if token_can_copy is False or token_idx != self.vocab.code.unk_id:
                        gen_token_mask = 1

                app_rule_idx_row.append(app_rule_idx)
                app_rule_mask_row.append(app_rule_mask)

                token_row.append(token_idx)
                gen_token_mask_row.append(gen_token_mask)
                copy_mask_row.append(copy_mask)

            self.apply_rule_idx_matrix.append(app_rule_idx_row)
            self.apply_rule_mask.append(app_rule_mask_row)

            self.primitive_idx_matrix.append(token_row)
            self.gen_token_mask.append(gen_token_mask_row)

            self.primitive_copy_mask.append(copy_mask_row)

        T = torch.cuda if self.cuda else torch
        self.apply_rule_idx_matrix = Variable(T.LongTensor(self.apply_rule_idx_matrix))
        self.apply_rule_mask = Variable(T.FloatTensor(self.apply_rule_mask))
        self.primitive_idx_matrix = Variable(T.LongTensor(self.primitive_idx_matrix))
        self.gen_token_mask = Variable(T.FloatTensor(self.gen_token_mask))
        self.primitive_copy_mask = Variable(T.FloatTensor(self.primitive_copy_mask))
        self.primitive_copy_token_idx_mask = Variable(torch.from_numpy(self.primitive_copy_token_idx_mask))
        if self.cuda:
            self.primitive_copy_token_idx_mask = self.primitive_copy_token_idx_mask.cuda()

    @property
    def primitive_mask(self):
        return 1. - torch.eq(self.gen_token_mask + self.primitive_copy_mask, 0).float()

    @cached_property
    def src_sents_var(self):
        return nn_utils.to_input_variable(self.src_sents, self.vocab.source,
                                          cuda=self.cuda)

    @cached_property
    def src_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.src_sents_len,
                                                    cuda=self.cuda)

    @cached_property
    def token_pos_list(self):
        batch_src_token_to_pos_map = []
        for e_id, e in enumerate(self.examples):
            aggregated_primitive_tokens = OrderedDict()
            for token_pos, token in enumerate(e.src_sent):
                aggregated_primitive_tokens.setdefault(token, []).append(token_pos)
