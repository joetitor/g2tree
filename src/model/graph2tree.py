import argparse
import time
import pickle as pkl
import data_utils
import os
import time
import numpy as np
from tree import Tree
import re

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import graph_utils
import warnings

warnings.filterwarnings('ignore')

from graph_model import GraphEncoder
from torch import optim
import random

class LSTM(nn.Module):
    def __init__(self, opt):
        super(LSTM, self).__init__()
        self.opt = opt
        self.i2h = nn.Linear(opt.rnn_size, 4*opt.rnn_size)
        self.h2h = nn.Linear(opt.rnn_size, 4*opt.rnn_size)
        if opt.dropoutrec > 0:
            self.dropout = nn.Dropout(opt.dropoutrec)

    def forward(self, x, prev_c, prev_h):
        gates = self.i2h(x) + self.h2h(prev_h)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4,1)
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)
        if self.opt.dropoutrec > 0:
            cellgate = self.dropout(cellgate)
        cy = (forgetgate * prev_c) + (ingate * cellgate)
        hy = outgate * F.tanh(cy)
        return cy, hy
    
class Dec_LSTM(nn.Module):
    def __init__(self, opt):
        super(Dec_LSTM, self).__init__()
        self.opt = opt

        self.word_embedding_size = 300

        self.i2h = nn.Linear(self.word_embedding_size+opt.rnn_size, 4*opt.rnn_size)
        self.h2h = nn.Linear(opt.rnn_size, 4*opt.rnn_size)



        if opt.dropoutrec > 0:
            self.dropout = nn.Dropout(opt.dropoutrec)

    def forward(self, x, prev_c, prev_h, parent_h):
        input_cat = torch.cat((x, parent_h),1)
        gates = self.i2h(input_cat) + self.h2h(prev_h)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4,1)
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)
        if self.opt.dropoutrec > 0:
            cellgate = self.dropout(cellgate)
        cy = (forgetgate * prev_c) + (ingate * cellgate)
        hy = outgate * F.tanh(cy)
        return cy, hy

class DecoderRNN(nn.Module):
    def __init__(self, opt, input_size):
        super(DecoderRNN, self).__init__()
        self.opt = opt
        self.hidden_size = opt.rnn_size
        # self.word_embedding_size = self.hidden_size
        self.word_embedding_size = 300
        self.embedding = nn.Embedding(input_size, self.word_embedding_size, padding_idx=0)

        self.lstm = Dec_LSTM(self.opt)
        if opt.dropout > 0:
            self.dropout = nn.Dropout(opt.dropout)

    # def forward(self, input_src, prev_c, prev_h, parent_h, hidden_for_feed):
    def forward(self, input_src, prev_c, prev_h, parent_h):

        src_emb = self.embedding(input_src) # batch_size x src_length x emb_size
        if self.opt.dropout > 0:
            src_emb = self.dropout(src_emb)
        prev_cy, prev_hy = self.lstm(src_emb, prev_c, prev_h, parent_h)
        return prev_cy, prev_hy

class AttnUnit(nn.Module):
    def __init__(self, opt, output_size):
        super(AttnUnit, self).__init__()
        self.opt = opt
        self.hidden_size = opt.rnn_size
        self.separate_attention = False
        if self.separate_attention:
            self.linear_att = nn.Linear(3*self.hidden_size, self.hidden_size)
        else:
            self.linear_att = nn.Linear(2*self.hidden_size, self.hidden_size)

        self.linear_out = nn.Linear(self.hidden_size, output_size)
        if opt.dropout > 0:
            self.dropout = nn.Dropout(opt.dropout)

        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, enc_s_top, dec_s_top, enc_2):
        # enc_s_top: [batch_size, sentence_length, hidden_size]
        # dec_s_top: [batch_size, hidden_size]
        dot = torch.bmm(enc_s_top, dec_s_top.unsqueeze(2))
        # dot: [batch_size, sentence_length, 1]
        attention = self.softmax(dot.squeeze(2)).unsqueeze(2)
        # attention: [batch_size, sentence_length, 1]
        # enc_s_top.premute(0,2,1): [batch_size, hidden_size, sentence_length]
        enc_attention = torch.bmm(enc_s_top.permute(0,2,1), attention)
        # enc_attention: [batch_size, hidden_size, 1]

        if self.separate_attention:
            dot_2 = torch.bmm(enc_2, dec_s_top.unsqueeze(2))
            attention_2 = self.softmax(dot_2.squeeze(2)).unsqueeze(2)
            enc_attention_2 = torch.bmm(enc_2.permute(0,2,1), attention_2)

        if self.separate_attention:
            hid = F.tanh(self.linear_att(torch.cat((enc_attention.squeeze(2), enc_attention_2.squeeze(2),dec_s_top), 1)))
        else:
            hid = F.tanh(self.linear_att(torch.cat((enc_attention.squeeze(2),dec_s_top), 1)))
        h2y_in = hid
        if self.opt.dropout > 0:
            h2y_in = self.dropout(h2y_in)
        h2y = self.linear_out(h2y_in)
        pred = self.logsoftmax(h2y)
        # return pred, h2y_in
        return pred

def eval_training(opt, train_loader, encoder, decoder, attention_decoder, encoder_optimizer, decoder_optimizer, attention_decoder_optimizer, criterion, using_gpu, word_manager, form_manager):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    attention_decoder_optimizer.zero_grad()
    enc_batch, enc_len_batch, dec_tree_batch = train_loader.random_batch()
    # print enc_batch

    enc_max_len = enc_len_batch

    enc_outputs = torch.zeros((opt.batch_size, enc_max_len, encoder.hidden_layer_dim), requires_grad=True)
    if using_gpu:
        enc_outputs = enc_outputs.cuda()
    
    # print enc_batch
    
    # print enc_batch
    fw_adj_info = torch.tensor(enc_batch['g_fw_adj'])
    bw_adj_info = torch.tensor(enc_batch['g_bw_adj'])
    feature_info = torch.tensor(enc_batch['g_ids_features'])
    batch_nodes = torch.tensor(enc_batch['g_nodes'])
    batch_wordlen = torch.tensor(enc_batch['word_len'])

    # node_embedding, graph_embedding, _ = encoder((fw_adj_info,bw_adj_info,feature_info,batch_nodes, batch_wordlen))
    # enc_outputs = node_embedding
    node_embedding, graph_embedding, structural_info = encoder((fw_adj_info,bw_adj_info,feature_info,batch_nodes, batch_wordlen))
    # print node_embedding.size()
    # print structural_info.size()

    # l1 = nn.Linear(2*opt.rnn_size, opt.rnn_size).cuda()
    # enc_outputs = l1(torch.cat([node_embedding, structural_info],2))
    # enc_outputs = (node_embedding+structural_info)/2
    enc_outputs = node_embedding
    # enc_outputs = torch.cat([node_embedding,structural_info], 1)

    graph_cell_state = torch.zeros((opt.batch_size, opt.rnn_size), dtype=torch.float, requires_grad=True)
    graph_hidden_state = torch.zeros((opt.batch_size, opt.rnn_size), dtype=torch.float, requires_grad=True)
    if using_gpu:
        graph_cell_state = graph_cell_state.cuda()
        graph_hidden_state = graph_hidden_state.cuda()
    
    graph_cell_state = graph_embedding
    graph_hidden_state = graph_embedding

    dec_s = {}
    for i in range(opt.dec_seq_length + 1):
        dec_s[i] = {}
        for j in range(opt.dec_seq_length + 1):
            dec_s[i][j] = {}

    # tree decode
    queue_tree = {}
    for i in range(1, opt.batch_size+1):
        queue_tree[i] = []
        queue_tree[i].append({"tree" : dec_tree_batch[i-1], "parent": 0, "child_index": 1})
    loss = 0
    cur_index, max_index = 1,1
    dec_batch = {}
    while (cur_index <= max_index):
        # build dec_batch for cur_index
        max_w_len = -1
        batch_w_list = []
        for i in range(1, opt.batch_size+1):
            w_list = []
            if (cur_index <= len(queue_tree[i])):
                t = queue_tree[i][cur_index - 1]["tree"]
                for ic in range (t.num_children):
                    if isinstance(t.children[ic], Tree):
                        w_list.append(4)
                        queue_tree[i].append({"tree" : t.children[ic], "parent" : cur_index, "child_index": ic + 1})
                    else:
                        w_list.append(t.children[ic])
                if len(queue_tree[i]) > max_index:
                    max_index = len(queue_tree[i])
            if len(w_list) > max_w_len:
                max_w_len = len(w_list)
            batch_w_list.append(w_list)
        dec_batch[cur_index] = torch.zeros((opt.batch_size, max_w_len + 2), dtype=torch.long)
        for i in range(opt.batch_size):
            w_list = batch_w_list[i]
            if len(w_list) > 0:
                for j in range(len(w_list)):
                    dec_batch[cur_index][i][j+1] = w_list[j]
                # add <S>, <E>
                if cur_index == 1:
                    dec_batch[cur_index][i][0] = 1
                else:
                    dec_batch[cur_index][i][0] = form_manager.get_symbol_idx('(')
                dec_batch[cur_index][i][len(w_list) + 1] = 2
        # initialize first decoder unit hidden state (zeros)
        if using_gpu:
            dec_batch[cur_index] = dec_batch[cur_index].cuda()
        # initialize using encoding results
        for j in range(1, 3):
            dec_s[cur_index][0][j] = torch.zeros((opt.batch_size, opt.rnn_size), dtype=torch.float, requires_grad=True)
            if using_gpu:
                dec_s[cur_index][0][j] = dec_s[cur_index][0][j].cuda()

        if cur_index == 1:
            for i in range(opt.batch_size):
                # dec_s[1][0][1][i, :] = enc_s[enc_len_batch[i]][1][i, :]
                # dec_s[1][0][2][i, :] = enc_s[enc_len_batch[i]][2][i, :]
                dec_s[1][0][1][i, :] = graph_cell_state[i]
                dec_s[1][0][2][i, :] = graph_hidden_state[i]

        else:
            for i in range(1, opt.batch_size+1):
                if (cur_index <= len(queue_tree[i])):
                    par_index = queue_tree[i][cur_index - 1]["parent"]
                    child_index = queue_tree[i][cur_index - 1]["child_index"]
                    
                    dec_s[cur_index][0][1][i-1,:] = \
                        dec_s[par_index][child_index][1][i-1,:]
                    dec_s[cur_index][0][2][i-1,:] = dec_s[par_index][child_index][2][i-1,:]
        gold_string = " "
        parent_h = dec_s[cur_index][0][2]
        # parent_h = [graph_embedding, dec_s[cur_index][0][2]]
        for i in range(dec_batch[cur_index].size(1) - 1):
            # if i == 0:
            #     hidden_for_feed = torch.zeros((opt.batch_size, opt.rnn_size), dtype=torch.float, requires_grad=True)
            #     if using_gpu:
            #         hidden_for_feed = hidden_for_feed.cuda()
            dec_s[cur_index][i+1][1], dec_s[cur_index][i+1][2] = decoder(dec_batch[cur_index][:,i], dec_s[cur_index][i][1], dec_s[cur_index][i][2], parent_h)
            pred = attention_decoder(enc_outputs, dec_s[cur_index][i+1][2], structural_info)

            loss += criterion(pred, dec_batch[cur_index][:,i+1])
        cur_index = cur_index + 1

    loss = loss / opt.batch_size
    loss.backward()
    torch.nn.utils.clip_grad_value_(encoder.parameters(),opt.grad_clip)
    torch.nn.utils.clip_grad_value_(decoder.parameters(),opt.grad_clip)
    torch.nn.utils.clip_grad_value_(attention_decoder.parameters(),opt.grad_clip)
    encoder_optimizer.step()
    decoder_optimizer.step()
    attention_decoder_optimizer.step()
    return loss

def convert_to_string(idx_list, form_manager):
    w_list = []
    for i in range(len(idx_list)):
        w_list.append(form_manager.get_idx_symbol(int(idx_list[i])))
    return " ".join(w_list)

def do_generate(encoder, decoder, attention_decoder, graph_input, word_manager, form_manager, opt, using_gpu, checkpoint):
    # initialize the rnn state to all zeros
    
    prev_c  = torch.zeros((1, encoder.hidden_layer_dim), requires_grad=False)
    prev_h  = torch.zeros((1, encoder.hidden_layer_dim), requires_grad=False)
    if using_gpu:
        prev_c = prev_c.cuda()
        prev_h = prev_h.cuda()
    # reversed order
    # enc_w_list.append(word_manager.get_symbol_idx('<S>'))
    # enc_w_list.insert(0, word_manager.get_symbol_idx('<E>'))
    # end = len(enc_w_list)
    graph_size = len(graph_input['g_nodes'][0])
    enc_outputs = torch.zeros((1, graph_size, encoder.hidden_layer_dim), requires_grad=False)
    if using_gpu:
        enc_outputs = enc_outputs.cuda()
    # for i in range(end-1, -1, -1):
    #     cur_input = torch.tensor(np.array(enc_w_list[i]), dtype=torch.long)
    #     if using_gpu:
    #         cur_input = cur_input.cuda()
    #     prev_c, prev_h = encoder(cur_input, prev_c, prev_h)
    #     enc_outputs[:, i, :] = prev_h

    # print graph_input['g_fw_adj']
    if graph_input['g_fw_adj'] == []:
        return "None"
    fw_adj_info = torch.tensor(graph_input['g_fw_adj'])
    bw_adj_info = torch.tensor(graph_input['g_bw_adj'])
    feature_info = torch.tensor(graph_input['g_ids_features'])
    batch_nodes = torch.tensor(graph_input['g_nodes'])
    batch_wordlen = torch.tensor(graph_input['word_len'])

    # node_embedding, graph_embedding, _ = encoder((fw_adj_info,bw_adj_info,feature_info,batch_nodes, batch_wordlen))
    # enc_outputs = node_embedding
    node_embedding, graph_embedding, structural_info = encoder((fw_adj_info,bw_adj_info,feature_info,batch_nodes, batch_wordlen))
    # l2 = nn.Linear(2*opt.rnn_size, opt.rnn_size).cuda()
    # enc_outputs = l2(torch.cat([node_embedding, structural_info],2))
    # enc_outputs = (node_embedding+structural_info)/2
    enc_outputs = node_embedding
    # enc_outputs = torch.cat([node_embedding,structural_info], 1)


    prev_c = graph_embedding
    prev_h = graph_embedding

    # decode
    queue_decode = []
    queue_decode.append({"s": (prev_c, prev_h), "parent":0, "child_index":1, "t": Tree()})
    head = 1
    while head <= len(queue_decode) and head <=100:
        s = queue_decode[head-1]["s"]
        parent_h = s[1]
        t = queue_decode[head-1]["t"]
        if head == 1:
            prev_word = torch.tensor([form_manager.get_symbol_idx('<S>')], dtype=torch.long)
        else:
            prev_word = torch.tensor([form_manager.get_symbol_idx('(')], dtype=torch.long)
        if using_gpu:
            prev_word = prev_word.cuda()
        i_child = 1
        while True:
            # curr_c, curr_h = decoder(prev_word, s[0], s[1], [graph_embedding, parent_h])

            # if i_child == 1:
            #     hidden_for_feed = torch.zeros((1, opt.rnn_size), dtype=torch.float, requires_grad=False)
            #     if using_gpu:
            #         hidden_for_feed = hidden_for_feed.cuda()

            # print s[0].size()
            curr_c, curr_h = decoder(prev_word, s[0], s[1], parent_h)
            prediction = attention_decoder(enc_outputs, curr_h, structural_info)
            s = (curr_c, curr_h)
            _, _prev_word = prediction.max(1)
            prev_word = _prev_word
            if int(prev_word[0]) == form_manager.get_symbol_idx('<E>') or t.num_children >= checkpoint["opt"].dec_seq_length:
                break
            elif int(prev_word[0]) == form_manager.get_symbol_idx('<N>'):
                #print("we predicted N");exit()
                queue_decode.append({"s": (s[0].clone(), s[1].clone()), "parent": head, "child_index":i_child, "t": Tree()})
                t.add_child(int(prev_word[0]))
            else:
                t.add_child(int(prev_word[0]))
            i_child = i_child + 1
        head = head + 1
    # refine the root tree (TODO, what is this doing?)
    for i in range(len(queue_decode)-1, 0, -1):
        cur = queue_decode[i]
        queue_decode[cur["parent"]-1]["t"].children[cur["child_index"]-1] = cur["t"]
    return queue_decode[0]["t"].to_list(form_manager)

def main(opt):
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    managers = pkl.load( open("{}/map.pkl".format(opt.data_dir), "rb" ) )
    word_manager, form_manager = managers
    using_gpu = False
    if opt.gpuid > -1:
        using_gpu = True

    # encoder = EncoderRNN(opt, word_manager.vocab_size)
    encoder = GraphEncoder(opt, word_manager.vocab_size)
    decoder = DecoderRNN(opt, form_manager.vocab_size)
    attention_decoder = AttnUnit(opt, form_manager.vocab_size)
    
    if using_gpu:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        attention_decoder = attention_decoder.cuda()
    # init parameters
    for name, param in encoder.named_parameters():
        if param.requires_grad:
            init.uniform_(param, -opt.init_weight, opt.init_weight)
    for name, param in decoder.named_parameters():
        if param.requires_grad:
            init.uniform_(param, -opt.init_weight, opt.init_weight)
    for name, param in attention_decoder.named_parameters():
        if param.requires_grad:
            init.uniform_(param, -opt.init_weight, opt.init_weight)
    ##-- load data
    train_loader = data_utils.MinibatchLoader(opt, 'train', using_gpu, word_manager)

    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)

    ##-- start training
    step = 0
    epoch = 0
    optim_state = {"learningRate" : opt.learning_rate, "alpha" :  opt.decay_rate}
    # default to rmsprop
    if opt.opt_method == 0:
        print("using RMSprop")
        encoder_optimizer = optim.RMSprop(encoder.parameters(),  lr=optim_state["learningRate"], alpha=optim_state["alpha"])
        decoder_optimizer = optim.RMSprop(decoder.parameters(),  lr=optim_state["learningRate"], alpha=optim_state["alpha"])
        attention_decoder_optimizer = optim.RMSprop(attention_decoder.parameters(),  lr=optim_state["learningRate"], alpha=optim_state["alpha"])
    elif opt.opt_method == 2:
        print("using adam")
        encoder_optimizer = optim.Adam(encoder.parameters(),  lr=optim_state["learningRate"], weight_decay=0.000001)
        decoder_optimizer = optim.Adam(decoder.parameters(),  lr=optim_state["learningRate"])
        attention_decoder_optimizer = optim.Adam(attention_decoder.parameters(),  lr=optim_state["learningRate"])
    criterion = nn.NLLLoss(size_average=False, ignore_index=0)

    print("Starting training.")
    encoder.train()
    decoder.train()
    attention_decoder.train()
    iterations = opt.max_epochs * train_loader.num_batch
    start_time = time.time()
    restarted = False
    # TODO revert back after tests
    #iterations = 2
    best_val_acc = 0
    for i in range(iterations):
        epoch = i // train_loader.num_batch
        train_loss = eval_training(opt, train_loader, encoder, decoder, attention_decoder, encoder_optimizer, decoder_optimizer, attention_decoder_optimizer, criterion, using_gpu, word_manager, form_manager)
        #return
        #exponential learning rate decay
        if opt.opt_method == 0:
            if i % train_loader.num_batch == 0 and opt.learning_rate_decay < 1:
                if epoch >= opt.learning_rate_decay_after:
                    decay_factor = opt.learning_rate_decay
                    optim_state["learningRate"] = optim_state["learningRate"] * decay_factor #decay it
                    for param_group in encoder_optimizer.param_groups:
                        param_group['lr'] = optim_state["learningRate"]
                    for param_group in decoder_optimizer.param_groups:
                        param_group['lr'] = optim_state["learningRate"]
                    for param_group in attention_decoder_optimizer.param_groups:
                        param_group['lr'] = optim_state["learningRate"]
        if (epoch == opt.restart) and not restarted:
            restarted = True
            optim_state["learningRate"] = opt.learning_rate
            for param_group in encoder_optimizer.param_groups:
                param_group['lr'] = optim_state["learningRate"]
                param_group['momentum'] = 0
            for param_group in decoder_optimizer.param_groups:
                param_group['lr'] = optim_state["learningRate"]
                param_group['momentum'] = 0
            for param_group in attention_decoder_optimizer.param_groups:
                param_group['lr'] = optim_state["learningRate"]
                param_group['momentum'] = 0

        #on last & first iteration
        if i == iterations - 1 or i % opt.print_every == 0:
            checkpoint = {}
            checkpoint["encoder"] = encoder
            checkpoint["decoder"] = decoder
            checkpoint["attention_decoder"] = attention_decoder
            checkpoint["opt"] = opt
            checkpoint["i"] = i
            checkpoint["epoch"] = epoch
            torch.save(checkpoint, "{}/model_seq2seq".format(opt.checkpoint_dir))

        if i % opt.print_every == 0:
            end_time = time.time()
            print("{}/{}, train_loss = {}, time since last print = {}".format( i, iterations, train_loss, (end_time - start_time)/60))
            start_time = time.time()

         
            test_checkpoint = torch.load("checkpoint_dir/model_seq2seq")
            test_encoder = test_checkpoint["encoder"]
            test_decoder = test_checkpoint["decoder"]
            test_attention_decoder = test_checkpoint["attention_decoder"]

            # test_encoder.eval()
            # test_decoder.eval()
            # test_attention_decoder.eval()
            if i / opt.print_every > 0:
                # show accuracy in train & test dataset
                data = pkl.load(open("{}/test.pkl".format(opt.data_dir), "rb"))
                graph_test_list = graph_utils.read_graph_data("{}/graph.test".format(opt.data_dir))
                reference_list = []
                candidate_list = []
                add_acc = 0.0
                for i in range(len(data)):
                    # print("example {}\n".format(i))
                    x = data[i]
                    reference = x[1]
                    graph_batch = graph_utils.cons_batch_graph([graph_test_list[i]])
                    graph_input = graph_utils.vectorize_batch_graph(graph_batch, word_manager)

                    candidate = do_generate(test_encoder, test_decoder, test_attention_decoder, graph_input, word_manager, form_manager, args, using_gpu, test_checkpoint)
                    candidate = [int(c) for c in candidate]

                    num_left_paren = sum(1 for c in candidate if form_manager.idx2symbol[int(c)]== "(")
                    num_right_paren = sum(1 for c in candidate if form_manager.idx2symbol[int(c)]== ")")
                    diff = num_left_paren - num_right_paren
                    #print(diff)
                    if diff > 0:
                        for i in range(diff):
                            candidate.append(form_manager.symbol2idx[")"])
                    elif diff < 0:
                        candidate = candidate[:diff]
        
                    ref_str = convert_to_string(reference, form_manager)
                    cand_str = convert_to_string(candidate, form_manager)

                    
                    def define_queries_order(str1, str2):
                        c_str = str2.replace(") ,", ") @")
                        c_list = c_str.strip().split("@")
                        flag = True
                        r_str = str1
                        for c_phrase in c_list:
                            if c_phrase.strip() in str1:
                                r_str = r_str.replace(c_phrase.strip(), "")
                            else:
                                flag = False
                        if len(r_str) != (r_str.count(",") + r_str.count(" ")):
                            flag = False
                        return flag
                    if (ref_str == cand_str) == False and define_queries_order(ref_str, cand_str) == True:
                        add_acc += 1.0
                    # def vague_equal(ref, cand):
                    #     lr = re.split('[()]', ref)
                    #     lc = re.split('[()]', cand)
                    #     for c_word in lc:
                    #         ref = ref.replace(c_word.strip(),"")
                    #     return len(ref) == (ref.count('(')) + (ref.count(')')) + (ref.count(' '))
                    # if (ref_str == cand_str) == False and vague_equal(ref_str, cand_str) == True:
                    #     add_acc += 1.0

                    reference_list.append(reference)
                    candidate_list.append(candidate)
                print add_acc/len(data)
                val_acc = data_utils.compute_tree_accuracy(candidate_list, reference_list, form_manager) + add_acc / len(data)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(test_checkpoint, "{}/best_model_seq2seq".format(opt.checkpoint_dir))
                print("ACCURACY = {}\n".format(val_acc))
            # test_encoder.train()
            # test_decoder.train()
            # test_attention_decoder.train()
        
        if train_loss != train_loss:
            print('loss is NaN.  This usually indicates a bug.')
            break
    print "best_acc: ",best_val_acc


if __name__ == "__main__":
    start = time.time()
    main_arg_parser = argparse.ArgumentParser(description="parser")
    main_arg_parser.add_argument('-gpuid', type=int, default=0, help='which gpu to use. -1 = use CPU')
    main_arg_parser.add_argument('-data_dir', type=str, default='../dataset/', help='data path')
    main_arg_parser.add_argument('-seed',type=int,default=123,help='torch manual random number generator seed')
    main_arg_parser.add_argument('-checkpoint_dir',type=str, default= 'checkpoint_dir', help='output directory where checkpoints get written')
    main_arg_parser.add_argument('-savefile',type=str, default='save',help='filename to autosave the checkpont to. Will be inside checkpoint_dir/')
    main_arg_parser.add_argument('-print_every',type=int, default=100,help='how many steps/minibatches between printing out the loss')
    main_arg_parser.add_argument('-rnn_size', type=int,default=500, help='size of LSTM internal state')
    main_arg_parser.add_argument('-num_layers', type=int, default=1, help='number of layers in the LSTM')
    main_arg_parser.add_argument('-dropout',type=float, default=0.4,help='dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
    main_arg_parser.add_argument('-dropoutrec',type=float,default=0.1,help='dropout for regularization, used after each c_i. 0 = no dropout')
    main_arg_parser.add_argument('-dropoutagg',type=float,default=0,help='dropout for regularization, used after each aggregator. 0 = no dropout')
    main_arg_parser.add_argument('-enc_seq_length',type=int, default=60,help='number of timesteps to unroll for')
    main_arg_parser.add_argument('-dec_seq_length',type=int, default=220,help='number of timesteps to unroll for')
    main_arg_parser.add_argument('-batch_size',type=int, default=20,help='number of sequences to train on in parallel')
    #main_arg_parser.add_argument('-batch_size',type=int, default=2,help='number of sequences to train on in parallel')
    main_arg_parser.add_argument('-max_epochs',type=int, default=400,help='number of full passes through the training data')
    main_arg_parser.add_argument('-opt_method', type=int,default=2,help='optimization method: 0-rmsprop 1-sgd 2-adam')
    main_arg_parser.add_argument('-learning_rate',type=float, default=0.001,help='learning rate')
    main_arg_parser.add_argument('-init_weight',type=float, default=0.08,help='initailization weight')
    main_arg_parser.add_argument('-learning_rate_decay',type=float, default=0.985,help='learning rate decay')
    main_arg_parser.add_argument('-learning_rate_decay_after',type=int, default=5,help='in number of epochs, when to start decaying the learning rate')
    main_arg_parser.add_argument('-restart',type=int, default=-1,help='in number of epochs, when to restart the optimization')
    main_arg_parser.add_argument('-decay_rate',type=float, default=0.95,help='decay rate for rmsprop')
    main_arg_parser.add_argument('-grad_clip',type=int, default=5,help='clip gradients at this value')

    # some arguments of graph encoder
    main_arg_parser.add_argument('-graph_encode_direction',type=str, default='bi',help='graph encode direction: bi or uni')
    main_arg_parser.add_argument('-sample_size_per_layer',type=int, default=6,help='sample_size_per_layer')
    main_arg_parser.add_argument('-sample_layer_size',type=int, default=6,help='sample_layer_size')
    main_arg_parser.add_argument('-concat',type=bool, default=True,help='concat in aggregators settings')

    args = main_arg_parser.parse_args()
    main(args)
    end = time.time()
    print("total time: {} minutes\n".format((end - start)/60))
