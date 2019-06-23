import argparse
import copy
import data_utils
from graph2tree import *
import torch
from tree import Tree
import graph_utils
import warnings

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="parser")
    main_arg_parser.add_argument('-gpuid', type=int, default=0, help='which gpu to use. -1 = use CPU')
    main_arg_parser.add_argument('-temperature', type=int, default=1, help='temperature of sampling')
    main_arg_parser.add_argument('-sample', type=int, default=0, help='0 to use max at each timestep (-beam_size=1), 1 to sample at each timestep, 2 to beam search')
    main_arg_parser.add_argument('-beam_size', type=int, default=20, help='beam size')
    main_arg_parser.add_argument('-display', type=int, default=1, help='whether display on console')
    main_arg_parser.add_argument('-data_dir', type=str, default='../dataset/', help='data path')
    main_arg_parser.add_argument('-input', type=str, default='test.t7', help='input data filename')
    main_arg_parser.add_argument('-output', type=str, default='output/seq2seq_output.txt', help='input data filename')
    main_arg_parser.add_argument('-model', type=str, default='checkpoint_dir/best_model_seq2seq', help='model checkpoint to use for sampling')
    main_arg_parser.add_argument('-seed',type=int,default=123,help='torch manual random number generator seed')

    # parse input params
    args = main_arg_parser.parse_args()
    # TODO, if the encoder was trained on a GPU do I need to call cuda
    using_gpu = False
    if args.gpuid > -1:
        using_gpu = True
    # load the model checkpoint
    checkpoint = torch.load(args.model)
    encoder = checkpoint["encoder"]
    decoder = checkpoint["decoder"]
    attention_decoder = checkpoint["attention_decoder"]
    # put in eval mode for dropout
    # encoder.eval()
    # decoder.eval()
    # attention_decoder.eval()
    encoder.train()
    decoder.train()
    attention_decoder.train()
    # initialize the vocabulary manager to display text
    managers = pkl.load( open("{}/map.pkl".format(args.data_dir), "rb" ) )
    word_manager, form_manager = managers
    # load data
    data = pkl.load(open("{}/test.pkl".format(args.data_dir), "rb"))
    graph_test_list = graph_utils.read_graph_data("{}/graph.test".format(args.data_dir))
    
    reference_list = []
    candidate_list = []
    add_acc = 0.0
    with open(args.output, "w") as output:
        # TODO change when running full -- this is to just reproduce the error
        #for i in range(30,50):
        #for i in range(278,280):
        for i in range(len(data)):
            # print("example {}\n".format(i))
            x = data[i]
            reference = x[1]
            graph_batch = graph_utils.cons_batch_graph([graph_test_list[i]])
            # print graph_batch
            graph_input = graph_utils.vectorize_batch_graph(graph_batch, word_manager)
            # print graph_input
            # break 
            candidate = do_generate(encoder, decoder, attention_decoder, graph_input, word_manager, form_manager, args, using_gpu, checkpoint)
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
            # print to console
            if args.display > 0:
                if ref_str != cand_str:
                    print "index: ", i
                    print(" ".join(x[0]))
                    print(ref_str)
                    print(cand_str)
                # print(' ')
                pass
            output.write("{}\n".format(ref_str))
            output.write("{}\n".format(cand_str))
        print add_acc/len(data)
        val_acc = data_utils.compute_tree_accuracy(candidate_list, reference_list, form_manager) + add_acc/len(data)
        print("ACCURACY = {}\n".format(val_acc))
        output.write("ACCURACY = {}\n".format(val_acc))
