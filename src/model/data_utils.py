import random
import math
from random import randint
import pickle as pkl
import numpy as np
import torch
import tree
import graph_utils
from operator import itemgetter

class SymbolsManager():
    def __init__(self, whether_add_special_tags):
        self.symbol2idx = {}
        self.idx2symbol = {}
        self.vocab_size = 0
        self.whether_add_special_tags = whether_add_special_tags
        if whether_add_special_tags:
            # PAD: padding token = 0
            self.add_symbol('<P>')
            # GO: start token = 1
            self.add_symbol('<S>')
            # EOS: end token = 2
            self.add_symbol('<E>')
            # UNK: unknown token = 3
            self.add_symbol('<U>')
            # NON: non-terminal token = 4
            self.add_symbol('<N>')

    def add_symbol(self,s):
        if s not in self.symbol2idx:
            self.symbol2idx[s] = self.vocab_size
            self.idx2symbol[self.vocab_size] = s
            self.vocab_size += 1
        return self.symbol2idx[s]
    
    def get_symbol_idx(self,s):
        if s not in self.symbol2idx:
            if self.whether_add_special_tags:
                return self.symbol2idx['<U>']
            else:
                print("not reached!")
                return 0
        return self.symbol2idx[s]

    def get_idx_symbol(self, idx):
        if idx not in self.idx2symbol:
            return '<U>'
        return self.idx2symbol[idx]
    
    def init_from_file(self, fn, min_freq, max_vocab_size):
        # the vocab file is sorted by word_freq
        print "loading vocabulary file: {}".format(fn)
        with open(fn,"r") as f:
            for line in f:
                l_list = line.strip().split('\t')
                c = int(l_list[1])
                if c >= min_freq:
                    self.add_symbol(l_list[0])
                if self.vocab_size > max_vocab_size:
                    break

    def get_symbol_idx_for_list(self,l):
        r = []
        for i in range(len(l)):
            # if l[i] not in [r'\+']:
                r.append(self.get_symbol_idx(l[i]))
        return r
    
class MinibatchLoader():
    def __init__(self, opt, mode, using_gpu, word_manager):
        data = pkl.load(open("{}/{}.pkl".format(opt.data_dir, mode), "rb" ))
        # if len(data) % opt.batch_size != 0:
        #     n = len(data)
        #     for i in range(len(data)%opt.batch_size):
        #         data.insert(n-i-1, data[n-i-1])
        graph_data = graph_utils.read_graph_data("{}/graph.{}".format(opt.data_dir, mode))
        # if len(graph_data) % opt.batch_size != 0:
        #     n = len(graph_data)
        #     for i in range(len(graph_data)%opt.batch_size):
        #         graph_data.insert(n-i-1, graph_data[n-i-1])

        self.enc_batch_list = []
        self.enc_len_batch_list = []
        self.dec_batch_list = []
        # used for copy
        # self.enc_for_copy_list = []
        p = 0
        while p + opt.batch_size <= len(data):
            # build encoder matrix

            batch_graph = [graph_data[p + idx] for idx in range(opt.batch_size)]
            combine_batch_graph = graph_utils.cons_batch_graph(batch_graph)
            vector_batch_graph = graph_utils.vectorize_batch_graph(combine_batch_graph,word_manager)

            self.enc_batch_list.append(vector_batch_graph)
            self.enc_len_batch_list.append(len(batch_graph[0]['g_ids']))
            # build decoder matrix
            tree_batch = []
            for i in range(opt.batch_size):
                tree_batch.append(data[p+i][2])
            self.dec_batch_list.append(tree_batch)
            p += opt.batch_size
        self.num_batch = len(self.enc_batch_list)
        assert(len(self.enc_batch_list) == len(self.dec_batch_list))

    def random_batch(self):
        p = randint(0,self.num_batch-1)
        return self.enc_batch_list[p], self.enc_len_batch_list[p], self.dec_batch_list[p]

    def all_batch(self):
        r = []
        for p in range(self.num_batch):
            r.append([self.enc_batch_list[p], self.enc_len_batch_list[p], self.dec_batch_list[p]])
        return r

class MinibatchLoader_for_seq2seq():
    def __init__(self, opt, mode, using_gpu, word_manager):
        data = pkl.load(open("{}/{}.pkl".format(opt.data_dir, mode), "rb" ))
        if len(data) % opt.batch_size != 0:
            n = len(data)
            for i in range(len(data)%opt.batch_size):
                data.insert(n-i-1, data[n-i-1])
        graph_data = graph_utils.read_graph_data("{}/graph.{}".format(opt.data_dir, mode))
        if len(graph_data) % opt.batch_size != 0:
            n = len(graph_data)
            for i in range(len(graph_data)%opt.batch_size):
                graph_data.insert(n-i-1, graph_data[n-i-1])

        self.enc_batch_list = []
        self.enc_len_batch_list = []
        self.dec_batch_list = []
        p = 0
        while p + opt.batch_size <= len(data):
            # build encoder matrix

            batch_graph = [graph_data[p + idx] for idx in range(opt.batch_size)]
            combine_batch_graph = graph_utils.cons_batch_graph(batch_graph)
            vector_batch_graph = graph_utils.vectorize_batch_graph(combine_batch_graph,word_manager)

            self.enc_batch_list.append(vector_batch_graph)
            self.enc_len_batch_list.append(len(batch_graph[0]['g_ids']))
            # build decoder matrix
            max_len = -1
            for i in range(opt.batch_size):
                w_list = data[p+i][1]
                if len(w_list) > max_len:
                    max_len = len(w_list)
            m_text = torch.zeros((opt.batch_size, max_len + 2), dtype=torch.long)
            if using_gpu:
                m_text = m_text.cuda()
            # add <S>
            m_text[:,0] = 1
            for i in range(opt.batch_size):
                w_list = data[p+i][1]
                for j in range(len(w_list)):
                    m_text[i][j+1] = w_list[j]
                # add <E>
                m_text[i][len(w_list)+1] = 2
            self.dec_batch_list.append(m_text)
            p += opt.batch_size
        self.num_batch = len(self.enc_batch_list)
        assert(len(self.enc_batch_list) == len(self.dec_batch_list))

    def random_batch(self):
        p = randint(0,self.num_batch-1)
        return self.enc_batch_list[p], self.enc_len_batch_list[p], self.dec_batch_list[p]

    def all_batch(self):
        r = []
        for p in range(self.num_batch):
            r.append([self.enc_batch_list[p], self.enc_len_batch_list[p], self.dec_batch_list[p]])
        return r

def convert_to_tree(r_list, i_left, i_right, form_manager):
    t = tree.Tree()
    level = 0
    left = -1
    for i in range(i_left, i_right):
        if r_list[i] == form_manager.get_symbol_idx('('):
            if level == 0:
                left = i
            level = level + 1
        elif r_list[i] == form_manager.get_symbol_idx(')'):
            level = level -1
            if level == 0:
                if i == left+1:
                    c = r_list[i]
                else:
                    c = convert_to_tree(r_list, left + 1, i, form_manager)
                t.add_child(c)
        elif level == 0:
            t.add_child(r_list[i])
    return t

def norm_tree(r_list, form_manager):
    q = [convert_to_tree(r_list, 0, len(r_list), form_manager)]
    head = 0
    while head < len(q):
        t = q[head]
        if t.num_children > 0:
            if (t.children[0] == form_manager.get_symbol_idx('and')) or (t.children[0] == form_manager.get_symbol_idx('or')):
                k = []
                for i in range(1, len(t.children)):
                    if isinstance(t.children[i], tree.Tree):
                        k.append((t.children[i].to_string(), i))
                    else:
                        k.append((str(t.children[i]), i))
                sorted_t_dict = []
                k.sort(key=itemgetter(0))
                for key1 in k:
                    sorted_t_dict.append(t.children[key1[1]])
                for i in range(t.num_children-1):
                    t.children[i+1] = \
                        sorted_t_dict[i]
        for i in range(len(t.children)):
            if isinstance(t.children[i], tree.Tree):
                q.append(t.children[i])

        head = head + 1
    return q[0]

def is_all_same(c1, c2):
    if len(c1) == len(c2):
        all_same = True
        for j in range(len(c1)):
            if c1[j] != c2[j]:
                all_same = False
                break
        return all_same
    else:
        return False

def compute_accuracy(candidate_list, reference_list):
    if len(candidate_list) != len(reference_list):
        print("candidate list has length {}, reference list has length {}\n".format(len(candidate_list), len(reference_list)))

    len_min = min(len(candidate_list), len(reference_list))
    c = 0
    for i in range(len_min):
        # print(candidate_list[i])
        # print(reference_list[i])
        # print ""
        if is_all_same(candidate_list[i], reference_list[i]):
            # print("above was all same")
            c = c+1
        else:
            pass
            # print(candidate_list[i])
            # print(reference_list[i])
            # print "wrong:", i
    return c/float(len_min)


def compute_tree_accuracy(candidate_list_, reference_list_, form_manager):
    candidate_list = []
    for i in range(len(candidate_list_)):
        #print("candidate\n\n")
        candidate_list.append(norm_tree(candidate_list_[i], form_manager).to_list(form_manager))
    reference_list = []
    for i in range(len(reference_list_)):
        reference_list.append(norm_tree(reference_list_[i], form_manager).to_list(form_manager))
    return compute_accuracy(candidate_list, reference_list)
