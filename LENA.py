import argparse
import math
import os.path
import timeit
from multiprocessing import JoinableQueue, Queue, Process

import numpy as np
import tensorflow as tf
import random
from datetime import datetime

class LENA:
    @property
    def n_entity(self):
        return self.__n_entity
    @property
    def n_relation(self):
        return self.__n_relation
    @property
    def H(self):
        return self.__H
    @property
    def L(self):
        return self.__L

    @property
    def n_train(self):
        return self.__train_triple.shape[0]

    @property
    def trainable_variables(self):
        return self.__trainable

    @property
    def hr_t(self):
        return self.__hr_t
    @property
    def train_hr_t(self):
        return self.__train_hr_t

    @property
    def ent_embedding(self):
        return self.__ent_embedding

    @property
    def rel_embedding(self):
        return self.__rel_embedding
    @property
    def entity_to_edge(self):
        return self.__entity_to_edge
    @property
    def edge_to_all_H(self):
        return self.__edge_to_all_H

    def raw_training_data(self, batch_size=100):
        n_triple = len(self.__train_triple)
        rand_idx = np.random.permutation(n_triple)

        start = 0
        while start < n_triple:
            end = min(start + batch_size, n_triple)
            yield self.__train_triple[rand_idx[start:end]]
            start = end

    def testing_data(self, batch_size=100):
        n_triple = len(self.__test_triple)
        start = 0
        while start < n_triple:
            end = min(start + batch_size, n_triple)
            yield self.__test_triple[start:end, :]
            start = end

    def validation_data(self, batch_size=100):
        n_triple = len(self.__valid_triple)
        start = 0
        while start < n_triple:
            end = min(start + batch_size, n_triple)
            yield self.__valid_triple[start:end, :]
            start = end
    def __init__(self, data_dir, embed_dim=100, combination_method='simple', dropout=0.5, neg_weight=0.5 , window_H = 2, window_L = 10):

        if combination_method.lower() not in ['simple', 'matrix']:
            raise NotImplementedError("LENA does not support using %s as combination method." % combination_method)

        self.__combination_method = combination_method

        self.__embed_dim = embed_dim
        self.__initialized = False
        self.__H = window_H
        self.__L = window_L

        self.__trainable = list()
        self.__dropout = dropout

        with open(os.path.join(data_dir, 'entity2id.txt'), 'r') as f:
            self.__n_entity = len(f.readlines())

        with open(os.path.join(data_dir, 'entity2id.txt'), 'r') as f:
            self.__entity_id_map = {x.strip().split('\t')[0]: int(x.strip().split('\t')[1]) for x in f.readlines()}
            self.__id_entity_map = {v: k for k, v in self.__entity_id_map.items()}

        print("N_ENTITY: %d" % self.__n_entity)

        with open(os.path.join(data_dir, 'relation2id.txt'), 'r') as f:
            self.__n_relation = len(f.readlines())

        with open(os.path.join(data_dir, 'relation2id.txt'), 'r') as f:
            self.__relation_id_map = {x.strip().split('\t')[0]: int(x.strip().split('\t')[1]) for x in f.readlines()}
            self.__id_relation_map = {v: k for k, v in self.__relation_id_map.items()}

        print("N_RELATION: %d" % self.__n_relation)

        def load_triple(file_path):
            with open(file_path, 'r') as f_triple:
                return np.asarray([[self.__entity_id_map[x.strip().split('\t')[0]],
                                    self.__entity_id_map[x.strip().split('\t')[1]],
                                    self.__relation_id_map[x.strip().split('\t')[2]]] for x in f_triple.readlines()],
                                  dtype=np.int32)

        def gen_hr_t(triple_data):
            hr_t = dict()
            for h, t, r in triple_data:
                if h not in hr_t:
                    hr_t[h] = dict()
                if r not in hr_t[h]:
                    hr_t[h][r] = set()
                hr_t[h][r].add(t)

            return hr_t
        def reciprocal_triple(data):
            new_data = []
            for triple in data:
                reciprocal = np.asarray([triple[1],triple[0],triple[2]+self.__n_relation])
                new_data.append(triple)
                new_data.append(reciprocal)
            return np.asarray(new_data)

        self.__train_triple = load_triple(os.path.join(data_dir, 'train.txt'))
        self.__train_triple = reciprocal_triple(self.__train_triple)
        print("N_TRAIN_TRIPLES: %d" % self.__train_triple.shape[0])

        self.__test_triple = load_triple(os.path.join(data_dir, 'test.txt'))
        self.__test_triple = reciprocal_triple(self.__test_triple)
        print("N_TEST_TRIPLES: %d" % self.__test_triple.shape[0])

        self.__valid_triple = load_triple(os.path.join(data_dir, 'valid.txt'))
        self.__valid_triple = reciprocal_triple(self.__valid_triple)
        print("N_VALID_TRIPLES: %d" % self.__valid_triple.shape[0])

        self.__train_hr_t = gen_hr_t(self.__train_triple)
        self.__test_hr_t = gen_hr_t(self.__test_triple)

        self.__hr_t = gen_hr_t(np.concatenate([self.__train_triple, self.__test_triple, self.__valid_triple], axis=0))
        ######################################################################################################################
        self.__entity_to_edge = {k: []for k in range(self.__n_entity)}
        for triple in self.__train_triple:
            self.__entity_to_edge[triple[1]].append(triple)
        for key,value in self.__entity_to_edge.items():
            self.__entity_to_edge[key] = np.asarray(self.__entity_to_edge[key])

        self.__edge_to_all_H = 0
        ######################################################################################################################
        bound = 6 / math.sqrt(embed_dim)


        self.__ent_embedding = tf.get_variable("ent_embedding", [self.__n_entity, embed_dim],
                                               initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                         maxval=bound,
                                                                                         seed=345))
        self.__trainable.append(self.__ent_embedding)

        self.__rel_embedding = tf.get_variable("rel_embedding", [self.__n_relation * 2, embed_dim],
                                               initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                         maxval=bound,
                                                                                         seed=346))
        
        self.__trainable.append(self.__rel_embedding)

        self.__Gamma_r = tf.get_variable("gamma_r",[self.__n_relation * 2,embed_dim],
            initializer = tf.random_uniform_initializer(minval=-bound,maxval=bound,seed=348))
        # self.__trainable.append(self.__Gamma_r)

        self.__Gamma_r_prim = tf.get_variable("gamma_r_prim",[self.__n_relation * 2,embed_dim],
            initializer = tf.random_uniform_initializer(minval=-bound,maxval=bound,seed=349))
        # self.__trainable.append(self.__Gamma_r_prim)

        self.__C = tf.get_variable("C",initializer=tf.ones([embed_dim]))

        self.__trainable.append(self.__C)

        self.__C_prim = tf.get_variable("C_prim",initializer=tf.ones([embed_dim]))

        self.__trainable.append(self.__C_prim)


        self.__hr_weighted_vector = tf.get_variable("simple_hr_combination_weights", [embed_dim * 2],
                                                    initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                              maxval=bound,
                                                                                              seed=445))
        self.__trainable.append(self.__hr_weighted_vector)
        self.__hr_combination_bias = tf.get_variable("combination_bias_hr",
                                                     initializer=tf.zeros([embed_dim]))

        self.__trainable.append(self.__hr_combination_bias)
        self.__hr_combination_bias_prim = tf.get_variable("combination_bias_hr_prim",
                                                     initializer=tf.zeros([embed_dim]))

        self.__trainable.append(self.__hr_combination_bias_prim)

    @staticmethod
    def __l1_normalize(x, dim, epsilon=1e-12, name=None):
        square_sum = tf.reduce_sum(tf.abs(x), [dim], keep_dims=True)
        x_inv_norm = tf.rsqrt(tf.maximum(square_sum, epsilon))
        return tf.mul(x, x_inv_norm, name=name)

    @staticmethod
    def sampled_softmax(tensor, weights):
        max_val = tf.reduce_max(tensor * tf.abs(weights), 1, keep_dims=True)
        tensor_rescaled = tensor - max_val
        tensor_exp = tf.exp(tensor_rescaled)
        tensor_sum = tf.reduce_sum(tensor_exp * tf.abs(weights), 1, keep_dims=True)

        return (tensor_exp / tensor_sum) * tf.abs(weights)  # all ignored elements will have a prob of 0.

    def train(self, inputs, regularizer_weight=1., scope=None):
        with tf.variable_scope(scope or type(self).__name__) as scp:
            if self.__initialized:
                scp.reuse_variables()         
            rel_embedding = self.__rel_embedding
            normalized_ent_embedding = self.__ent_embedding
            Gamma_r = self.__Gamma_r
            Gamma_r_prim = self.__Gamma_r_prim
            
            normalized_ent_embedding = tf.concat([normalized_ent_embedding,tf.zeros([1,self.__embed_dim])],axis = 0)
            rel_embedding = tf.concat([rel_embedding,tf.zeros([1,self.__embed_dim])],axis = 0)

            hr_tlist, hr_tlist_weight, hr_tlist_HLe = inputs

            # (?, dim)
            hr_tlist_h = tf.nn.embedding_lookup(normalized_ent_embedding, hr_tlist[:, 0])
            hr_tlist_r = tf.nn.embedding_lookup(rel_embedding, hr_tlist[:, 1])
            
            HLe_0 = tf.nn.embedding_lookup(normalized_ent_embedding, hr_tlist_HLe[:,:,:, 0])
            HLe_2 = tf.nn.embedding_lookup(rel_embedding, hr_tlist_HLe[:,:,:, 2])

            gamma_r_h = tf.nn.embedding_lookup(Gamma_r,hr_tlist[:, 1])
            gamma_r_h = tf.expand_dims(gamma_r_h , axis = 1)
            gamma_r_h = tf.expand_dims(gamma_r_h , axis = 1)
            SE_HLe = tf.reduce_sum(tf.multiply(HLe_2,gamma_r_h),axis = 3)
            self.AlphaE_HLe = tf.nn.softmax(SE_HLe)
            vaE_HLe = tf.reduce_sum(tf.multiply(HLe_0 , tf.expand_dims(self.AlphaE_HLe , axis = 3)),axis = 2)
            vaE = tf.reduce_max(vaE_HLe,axis = 1)

            gamma_r_r = tf.nn.embedding_lookup(Gamma_r_prim,hr_tlist[:, 1])
            gamma_r_r = tf.expand_dims(gamma_r_r , axis = 1)
            gamma_r_r = tf.expand_dims(gamma_r_r , axis = 1)
            SR_HLe = tf.reduce_sum(tf.multiply(HLe_2,gamma_r_r),axis = 3)
            self.AlphaR_HLe = tf.nn.softmax(SR_HLe)
            vaR_HLe = tf.reduce_sum(tf.multiply(HLe_2 , tf.expand_dims(self.AlphaR_HLe , axis = 3)),axis = 2)
            vaR = tf.reduce_max(vaR_HLe,axis = 1)


            # shape (?, dim)
            # hr_tlist_hr = vaE + hr_tlist_r 
            hr_tlist_hr = vaE + hr_tlist_r
            hr_tlist_hr_prim = hr_tlist_h + vaR

            hrt_res = tf.matmul(tf.nn.dropout(tf.tanh(hr_tlist_hr + self.__hr_combination_bias), self.__dropout),
                self.__ent_embedding * self.__C,transpose_b=True) + tf.matmul(
                tf.nn.dropout(tf.tanh(hr_tlist_hr_prim + self.__hr_combination_bias_prim), self.__dropout),
                self.__ent_embedding * self.__C_prim,transpose_b=True)

            self.regularizer_loss = regularizer_loss = tf.reduce_sum(tf.abs(self.__ent_embedding)) + tf.reduce_sum(
                tf.abs(self.__hr_combination_bias)) + tf.reduce_sum(tf.abs(self.__hr_combination_bias_prim)) + tf.reduce_sum(
                tf.abs(self.__rel_embedding))+ tf.reduce_sum(tf.abs(self.__Gamma_r))+ tf.reduce_sum(
                tf.abs(self.__C))+ tf.reduce_sum(tf.abs(self.__C_prim))+ tf.reduce_sum(tf.abs(self.__Gamma_r_prim))

            self.hrt_softmax = hrt_res_softmax = self.sampled_softmax(hrt_res, hr_tlist_weight)

            hrt_loss = -tf.reduce_sum(
                tf.log(tf.clip_by_value(hrt_res_softmax, 1e-10, 1.0)) * tf.maximum(0.,
                                                                                   hr_tlist_weight) / tf.reduce_sum(
                    tf.maximum(0., hr_tlist_weight), 1, keep_dims=True))

            return hrt_loss + regularizer_loss * regularizer_weight

    def test(self, inputs, test_HLe_input, scope=None):
        with tf.variable_scope(scope or type(self).__name__) as scp:
            scp.reuse_variables()

            test_ent_embedding = self.__ent_embedding
            test_ent_embedding = tf.concat([test_ent_embedding,tf.zeros([1,self.__embed_dim])],axis = 0)
            test_rel_embedding = self.__rel_embedding
            test_rel_embedding = tf.concat([test_rel_embedding,tf.zeros([1,self.__embed_dim])],axis = 0)
            h = tf.nn.embedding_lookup(self.__ent_embedding, inputs[:, 0])
            t = tf.nn.embedding_lookup(self.__ent_embedding, inputs[:, 1])
            r = tf.nn.embedding_lookup(self.__rel_embedding, inputs[:, 2])

            HLe_0 = tf.nn.embedding_lookup(test_ent_embedding, test_HLe_input[:,:,:, 0])
            HLe_2 = tf.nn.embedding_lookup(test_rel_embedding, test_HLe_input[:,:,:, 2])

            gamma_r_hr = tf.nn.embedding_lookup(self.__Gamma_r,inputs[:, 2])
            gamma_r_hr = tf.expand_dims(gamma_r_hr , axis = 1)
            gamma_r_hr = tf.expand_dims(gamma_r_hr , axis = 1)
            SE_HLe = tf.reduce_sum(tf.multiply(HLe_2,gamma_r_hr),axis = 3)
            AlphaE_HLe = tf.nn.softmax(SE_HLe)
            vaE_HLe = tf.reduce_sum(tf.multiply(HLe_0 , tf.expand_dims(AlphaE_HLe , axis = 3)),axis = 2)
            vaE = tf.reduce_max(vaE_HLe,axis = 1)

            gamma_r_r = tf.nn.embedding_lookup(self.__Gamma_r_prim,inputs[:, 2])
            gamma_r_r = tf.expand_dims(gamma_r_r , axis = 1)
            gamma_r_r = tf.expand_dims(gamma_r_r , axis = 1)
            SR_HLe = tf.reduce_sum(tf.multiply(HLe_2,gamma_r_r),axis = 3)
            AlphaR_HLe = tf.nn.softmax(SR_HLe)
            vaR_HLe = tf.reduce_sum(tf.multiply(HLe_2 , tf.expand_dims(AlphaR_HLe , axis = 3)),axis = 2)
            vaR = tf.reduce_max(vaR_HLe,axis = 1)

            ent_mat = tf.transpose(self.__ent_embedding * self.__C)
            ent_mat_prim = tf.transpose(self.__ent_embedding * self.__C_prim)

            # predict tails
            # hr = vaE + r
            hr = vaE + r
            hr_prim = h + vaR
            hrt_res = tf.matmul(tf.tanh(hr + self.__hr_combination_bias), ent_mat) + tf.matmul(
                tf.tanh(hr_prim + self.__hr_combination_bias_prim), ent_mat_prim)
            _, tail_ids = tf.nn.top_k(hrt_res, k=self.__n_entity)

            return tail_ids


def train_ops(model, learning_rate=0.1, optimizer_str='gradient', regularizer_weight=1.0):

    train_hrt_input = tf.placeholder(tf.int32, [None, 2])
    train_hrt_weight = tf.placeholder(tf.float32, [None, model.n_entity])
    train_hrt_HLe_input = tf.placeholder(tf.int32 , [None , model.H,model.L + 1 ,4])

    loss = model.train([train_hrt_input, train_hrt_weight,train_hrt_HLe_input],
                       regularizer_weight=regularizer_weight)
    if optimizer_str == 'gradient':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif optimizer_str == 'rms':
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    elif optimizer_str == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    else:
        raise NotImplementedError("Does not support %s optimizer" % optimizer_str)

    grads = optimizer.compute_gradients(loss, model.trainable_variables)

    op_train = optimizer.apply_gradients(grads)

    return train_hrt_input, train_hrt_weight, train_hrt_HLe_input, loss, op_train


def test_ops(model):

    test_input = tf.placeholder(tf.int32, [None, 3])
    test_HLe_input = tf.placeholder(tf.int32, [None, model.H,model.L + 1,4])
    tail_ids = model.test(test_input,test_HLe_input)

    return test_input, tail_ids, test_HLe_input


def worker_func(in_queue, out_queue, hr_t):
    while True:
        dat = in_queue.get()
        if dat is None:
            in_queue.task_done()
            continue
        testing_data, tail_pred = dat
        out_queue.put(test_evaluation(testing_data, tail_pred, hr_t))
        in_queue.task_done()


def data_generator_func(in_queue, out_queue, hr_t, n_entity, neg_weight,edge_to_all_H,n_relation,H,L,entity_to_edge):
    while True:
        dat = in_queue.get()
        if dat is None:
            break
        # [head(tail), relation, #of_total_positive_candidates, positive_instances..., negative_instances...]
        hr_tlist = list()
        hr_tweight = list()
        hr_tlist_HLe = list()

        htr = dat

        for idx in range(htr.shape[0]):
            hr_tweight.append(
                [1. if x in hr_t[htr[idx, 0]][htr[idx, 2]] else y for
                 x, y in enumerate(np.random.choice([0., -1.], size=n_entity, p=[1 - neg_weight, neg_weight]))])

            hr_tlist.append([htr[idx, 0], htr[idx, 2]])
            
            edge = htr[idx]
            if edge[2] >= n_relation:
                edge_reciprocal = np.asarray([edge[1],edge[0],edge[2] - n_relation])
            else:
                edge_reciprocal = np.asarray([edge[1],edge[0],edge[2] + n_relation])
            # He = edge_to_all_H[','.join(str(i) for i in edge)]
            itself_index_1 = np.where(np.all(entity_to_edge[edge[0]]==edge,axis=1))[0]
            itself_index_2 = np.where(np.all(entity_to_edge[edge[0]]==edge_reciprocal,axis=1))[0]
            itself_index = np.concatenate((itself_index_1,itself_index_2),axis = 0)
            He = np.delete(entity_to_edge[edge[0]],itself_index,axis = 0)
            HLe = np.ndarray(shape=[H,L + 1,4],dtype=np.int32)
            for i in range(H):
                H0 = np.asarray([edge[0],edge[0],edge[2],edge[2]])
                Hl = np.zeros(shape=[L , 4],dtype=np.int32)
                Hl[:,[0,1]] = n_entity
                Hl[:,[2,3]] = n_relation * 2
                if 0 <= len(He) < L:
                    for L_index in range(len(He)):
                        Hl[L_index,[0,1,2]] = He[L_index]
                        Hl[L_index,3] = edge[2]
                else:
                    index = np.asarray(random.sample([k for k in range(len(He))],L))
                    Hl[:,[0,1,2]] = He[index]
                    Hl[:,3] = edge[2]
                HLe[i] = np.row_stack((H0,Hl))
            hr_tlist_HLe.append(HLe)

        out_queue.put((np.asarray(hr_tlist, dtype=np.int32), np.asarray(hr_tweight, dtype=np.float32),
                       np.asarray(hr_tlist_HLe,dtype=np.int32)))


def test_evaluation(testing_data, tail_pred, hr_t):
    assert len(testing_data) == len(tail_pred)
    mean_rank_t = list()
    filtered_mean_rank_t = list()

    for i in range(len(testing_data)):
        h = testing_data[i, 0]
        t = testing_data[i, 1]
        r = testing_data[i, 2]

        mr = 1
        for val in tail_pred[i]:
            if val == t:
                mean_rank_t.append(mr)
            mr += 1

        fmr = 1
        for val in tail_pred[i]:
            if val == t:
                filtered_mean_rank_t.append(fmr)
                break
            if h in hr_t and r in hr_t[h] and val in hr_t[h][r]:
                continue
            else:
                fmr += 1

    return (mean_rank_t, filtered_mean_rank_t)

def HLe_TLe_generator(model,testing_data):
    HLe_batch = np.ndarray(shape = [np.shape(testing_data)[0],model.H,model.L+1,4],dtype = np.int32)
    batch_index = 0
    for edge in testing_data:
        He = model.entity_to_edge[edge[0]]
        HLe = np.ndarray(shape=[model.H,model.L + 1,4],dtype=np.int32)
        for i in range(model.H):
            H0 = np.asarray([edge[0],edge[0],edge[2],edge[2]])
            Hl = np.zeros(shape=[model.L , 4],dtype=np.int32)
            Hl[:,[0,1]] = model.n_entity
            Hl[:,[2,3]] = model.n_relation * 2
            if 0 <= len(He) < model.L:
                for L_index in range(len(He)):
                    Hl[L_index,[0,1,2]] = He[L_index]
                    Hl[L_index,3] = edge[2]
            else:
                index = np.asarray(random.sample([k for k in range(len(He))],model.L))
                Hl[:,[0,1,2]] = He[index]
                Hl[:,3] = edge[2]
            HLe[i] = np.row_stack((H0,Hl))
        HLe_batch[batch_index] = HLe       
        batch_index += 1
    return HLe_batch

def main(_):
    parser = argparse.ArgumentParser(description='LENA.')
    parser.add_argument('--data', dest='data_dir', type=str, help="Data folder", default='./data/FB15k/')
    parser.add_argument('--lr', dest='lr', type=float, help="Learning rate", default=0.01)
    parser.add_argument("--dim", dest='dim', type=int, help="Embedding dimension", default=200)
    parser.add_argument("--H", dest='H', type=int, help="window number", default=2)
    parser.add_argument("--L", dest='L', type=int, help="window length", default=10)
    parser.add_argument("--batch", dest='batch', type=int, help="Batch size", default=200)
    parser.add_argument("--comb", dest="combination_method", type=str, help="Combination method", default='simple')
    parser.add_argument("--worker", dest='n_worker', type=int, help="Evaluation worker", default=1)
    parser.add_argument("--generator", dest='n_generator', type=int, help="Data generator", default=1)
    parser.add_argument("--eval_batch", dest="eval_batch", type=int, help="Evaluation batch size", default=500)
    parser.add_argument("--save_dir", dest='save_dir', type=str, help="Model path", default='./')
    parser.add_argument("--load_model", dest='load_model', type=str, help="Model file", default="")
    parser.add_argument("--save_per", dest='save_per', type=int, help="Save per x iteration", default=3)
    parser.add_argument("--eval_per", dest='eval_per', type=int, help="Evaluate every x iteration", default=1)
    parser.add_argument("--max_iter", dest='max_iter', type=int, help="Max iteration", default=100)
    parser.add_argument("--summary_dir", dest='summary_dir', type=str, help="summary directory",
                        default='./LENA_summary/')
    parser.add_argument("--keep", dest='drop_out', type=float, help="Keep prob (1.0 keep all, 0. drop all)",
                        default=0.5)
    parser.add_argument("--optimizer", dest='optimizer', type=str, help="Optimizer", default='adam')
    parser.add_argument("--prefix", dest='prefix', type=str, help="model_prefix", default='DEFAULT')
    parser.add_argument("--loss_weight", dest='loss_weight', type=float, help="Weight on parameter loss", default=1e-5)
    parser.add_argument("--neg_weight", dest='neg_weight', type=float, help="Sampling weight on negative examples",
                        default=0.1)

    args = parser.parse_args()

    print(args)

    model = LENA(args.data_dir, embed_dim=args.dim, combination_method=args.combination_method,
                  dropout=args.drop_out, neg_weight=args.neg_weight ,window_H = args.H, window_L = args.L)

    train_hrt_input, train_hrt_weight,  \
    train_hrt_HLe_input,  \
    train_loss, train_op = train_ops(model, learning_rate=args.lr,
                                     optimizer_str=args.optimizer,
                                     regularizer_weight=args.loss_weight)
    test_input,  test_tail, test_HLe_input = test_ops(model)
    
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as session:
        tf.initialize_all_variables().run()

        saver = tf.train.Saver()

        iter_offset = 0

        if args.load_model is not None and os.path.exists(args.load_model):
            saver.restore(session, args.load_model)
            iter_offset = int(args.load_model.split('.')[-2].split('_')[-1]) + 1
            print("Load model from %s, iteration %d restored." % (args.load_model, iter_offset))

        total_inst = model.n_train

        # training data generator
        raw_training_data_queue = Queue()
        training_data_queue = Queue()
        data_generators = list()
        for i in range(args.n_generator):
            data_generators.append(Process(target=data_generator_func, args=(
                raw_training_data_queue, training_data_queue, model.train_hr_t, model.n_entity, args.neg_weight ,
                model.edge_to_all_H,model.n_relation,model.H,model.L,model.entity_to_edge)))
            data_generators[-1].start()

        evaluation_queue = JoinableQueue()
        result_queue = Queue()
        for i in range(args.n_worker):
            worker = Process(target=worker_func, args=(evaluation_queue, result_queue, model.hr_t))
            worker.start()


        for n_iter in range(iter_offset, args.max_iter):
            start_time = timeit.default_timer()
            accu_loss = 0.
            accu_re_loss = 0.
            ninst = 0

            print("initializing raw training data...")
            nbatches_count = 0
            for dat in model.raw_training_data(batch_size=args.batch):
                raw_training_data_queue.put(dat)
                nbatches_count += 1
            print("raw training data initialized.")

            alpha_head = []
            alpha_r = []
            while nbatches_count > 0:
                nbatches_count -= 1

                hr_tlist, hr_tweight, hr_tlist_HLe = training_data_queue.get()
                l, rl, _ ,alpha_h_HLe,alpha_r_HLe= session.run(
                    [train_loss, model.regularizer_loss, train_op, model.AlphaE_HLe,model.AlphaR_HLe], {train_hrt_input: hr_tlist,
                                                                     train_hrt_weight: hr_tweight,
                                                                     train_hrt_HLe_input: hr_tlist_HLe})

                alpha_head.extend(list(alpha_h_HLe[:,:,0].flatten()))
                alpha_r.extend(list(alpha_r_HLe[:,:,0].flatten()))
                

                accu_loss += l
                accu_re_loss += rl
                ninst += len(hr_tlist)

                if ninst % (5000) is not None:
                    print(
                        '[%d sec](%d/%d) : %.2f -- loss : %.5f rloss: %.5f ' % (
                            timeit.default_timer() - start_time, ninst, total_inst, float(ninst) / total_inst,
                            l / (len(hr_tlist)),
                            args.loss_weight * (rl / (len(hr_tlist)))))
            print ("alpha_head_mean:",float(sum(alpha_head)/len(alpha_head)))
            print ("alpha_r_mean:",float(sum(alpha_r)/len(alpha_r)))
            print("")
            print("iter %d avg loss %.5f, time %.3f" % (n_iter, accu_loss / ninst, timeit.default_timer() - start_time))

            if n_iter % args.save_per == 0 or n_iter == args.max_iter - 1:
                save_path = saver.save(session,
                                       os.path.join(args.save_dir,
                                                    "LENA_" + str(args.prefix) + "_" + str(n_iter) + ".ckpt"))
                print("Model saved at %s" % save_path)

            if n_iter % args.eval_per == 0 or n_iter == args.max_iter - 1:
                for data_func, test_type in zip([model.validation_data, model.testing_data], ['VALID', 'TEST']):
                    accu_mean_rank_t = list()
                    accu_filtered_mean_rank_t = list()

                    evaluation_count = 0

                    for testing_data in data_func(batch_size=args.eval_batch):
                        HLe = HLe_TLe_generator(model,testing_data)
                        tail_pred = session.run(test_tail,
                                                           {test_input: testing_data,
                                                           test_HLe_input: HLe})

                        evaluation_queue.put((testing_data,tail_pred))
                        evaluation_count += 1

                    for i in range(args.n_worker):
                        evaluation_queue.put(None)

                    print("waiting for worker finishes their work")
                    evaluation_queue.join()
                    print("all worker stopped.")
                    while evaluation_count > 0:
                        evaluation_count -= 1

                        (mrt, fmrt) = result_queue.get()
                        accu_mean_rank_t += mrt
                        accu_filtered_mean_rank_t += fmrt

                    print(
                        "[%s] ITER %d [TAIL PREDICTION] MEAN RANK: %.1f FILTERED MEAN RANK %.1f HIT@10 %.3f FILTERED HIT@10 %.3f" %
                        (test_type, n_iter, np.mean(accu_mean_rank_t), np.mean(accu_filtered_mean_rank_t),
                         np.mean(np.asarray(accu_mean_rank_t, dtype=np.int32) < 11),
                         np.mean(np.asarray(accu_filtered_mean_rank_t, dtype=np.int32) < 11)))
                    
                    # with open(test_type+'_rank_list_' + str(n_iter)+'.txt','w') as fout:
                    #     fout.writelines('\t'.join([str(i) for i in accu_mean_rank_t]))
                    # with open(test_type+'_filter_rank_list_' + str(n_iter)+'.txt','w') as fout:
                    #     fout.writelines('\t'.join([str(i) for i in accu_filtered_mean_rank_t]))
                    
                    print ('Mean Rank:',np.mean(np.asarray(accu_mean_rank_t)))
                    print ('filter Mean Rank:',np.mean(np.asarray(accu_filtered_mean_rank_t)))
                    print ('MRR:',np.mean(np.reciprocal(np.asarray(accu_mean_rank_t),dtype=np.float32)))
                    print ('filter MRR:',np.mean(np.reciprocal(np.asarray(accu_filtered_mean_rank_t),dtype=np.float32)))
                    print ('hit @10:',np.mean(np.asarray(accu_mean_rank_t, dtype=np.int32) < 11))
                    print ('filter hit @1:',np.mean(np.asarray(accu_filtered_mean_rank_t, dtype=np.int32) < 2))
                    print ('filter hit @3:',np.mean(np.asarray(accu_filtered_mean_rank_t, dtype=np.int32) < 4))
                    print ('filter hit @10:',np.mean(np.asarray(accu_filtered_mean_rank_t, dtype=np.int32) < 11))

                    print ('hit @1:',np.mean(np.asarray(accu_mean_rank_t, dtype=np.int32) < 2))
                    print ('hit @3:',np.mean(np.asarray(accu_mean_rank_t, dtype=np.int32) < 4))


if __name__ == '__main__':
    tf.app.run()
