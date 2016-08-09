#!/usr/bin/env python
from __future__ import print_function
import argparse
import sys

import chainer
from chainer import cuda
from chainer import serializers
import copy
import time
from datetime import datetime
import json
from multiprocessing import Pool
import math
import numpy as np
import six

from links.gru import GRU
from links.lstm import LSTM
from links.sumcontext_bilinear import SumContextBilinear
from links.skip_cnn import SkipCNN
from links.baseline import Baseline
from links.transfer import Transfer
from links.asymmetric import Asymmetric
from links.asymmetric_tanh import AsymmetricTanh
from links.asymmetric_tanh_deep import AsymmetricTanhDeep

parser = argparse.ArgumentParser()
parser.add_argument('--save-path-model', '--save-path', '-savem', default=None, type=str,
                    help='save path')
parser.add_argument('--data-path', '-dp', default="./", type=str,
                    help='data path')
parser.add_argument('--model-name', '-mn', default="model", type=str,
                    help='model name')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--n-epoch', '--epoch', '-e', default=10, type=int,
                    help='epoch')
parser.add_argument('--n-units', '--units', '-u', default=1200, type=int,
                    help='dimension size (# of units)')
parser.add_argument('--n-in-units', '--in-units', '-iu', default=4800, type=int,
                    help='dimension size (# of units)')
parser.add_argument('--per', '-p', default=8192, type=int,
                    help='result output timing (per PER iters)')
parser.add_argument('--batchsize', '-b', default=64, type=int,
                    help='size of minibatch')
parser.add_argument('--seed', '-s', default=None, type=int,
                    help='initail random seed')
parser.add_argument('--grad-clip', '-c', default=5, type=float,
                    help='glad clip number')
parser.add_argument('--d-ratio', '--dropout-ratio', '--dropout', '-d', default=0.1, type=float,
                    help='dropout ratio')
parser.add_argument('--init-range', '-ir', default=0.05, type=float,
                    help='init range')
parser.add_argument('--speed-up', '-su', default=-1, type=int,
                    help='highway mode. { -1:none, 1: skip validation at 1-6th epoch, 2: and 7-30th epoch skip validation at even number epoch }')
parser.add_argument('--start-lr', '-lr', default=1e-4, type=float,
                    help='starting learning rate for decayed-sgd')
parser.add_argument('--decay-rate', '-dr', default=1.2, type=float,
                    help='decaying rate (1/DR) of learning rate for decayed-sgd')
parser.add_argument('--weight-decay', '-wd', default=0., type=float,
                    help='weight decaying rate ((1-WD)/1). e.g. 1e-5')
parser.add_argument('--save-epoch', '-se', default="100", type=str,
                    help='save timing: e.g "31,34,37" ')
parser.add_argument('--n-pool', '--pool', '-np', default=-1, type=int,
                    help='# of parallel pools used for cpu multi processing')
parser.add_argument('--frac-eval', '-fe', default=1, type=int,
                    help='evaluation timing. per 1/FE epoch.')
parser.add_argument('--w2v-path', '-w2v', default=None, type=str,
                    help='w2v path')
parser.add_argument('--load-model', '-lm', default=None, type=str,
                    help='loading pretrained model path')
parser.add_argument('--load-corpus', default=None, type=str,
                    help='load corpus.pkl path')
parser.add_argument('--model-type', '-mt', default="gru", type=str,
                    help='model type')
parser.add_argument('--margin', '-margin', default=1., type=float,
                    help='margin')
parser.add_argument('--onebyone', dest='onebyone', action='store_true')
parser.add_argument('--no-onebyone', dest='onebyone', action='store_false')
parser.add_argument('--negative-type', '-nt', default=-1, type=int,
                    help='negative instance type. if NT<=0: use only other 5th. if 1<=NT<=4: use own 1-4th incrementally. if 5<=NT: use own 1-4th but also other 5th with ((NT-4)/NT)%')
parser.set_defaults(onebyone=False)

args = parser.parse_args()
xp = cuda.cupy if args.gpu >= 0 else np
chainer.Function.type_check_enable = False

if args.save_path_model:
    args.save_path_model = args.save_path_model.rstrip("/")+"/"
args.data_path = args.data_path.rstrip("/")+"/"

if args.seed is None:
    args.seed = np.random.randint(100000)# random seed


def load_processed_dataset(load_corpus):
    args.load_corpus = args.load_corpus.rstrip("/") + "/"
    print(' load json')
    prev = time.time()
    train_data = json.load(open(load_corpus + "train_data.json"))
    valid_data = json.load(open(load_corpus + "valid_data.json"))
    test_data = json.load(open(load_corpus + "test_data.json"))
    print('', time.time()-prev, 's')
    prev = time.time()

    print(' load vectors')
    train_vectors = np.load(load_corpus + "train_vectors.npy").astype(np.float32)
    valid_vectors = np.load(load_corpus + "valid_vectors.npy").astype(np.float32)
    test_vectors = np.load(load_corpus + "test_vectors.npy").astype(np.float32)
    print('', time.time()-prev, 's')
    prev = time.time()

    print(' set vectors train')
    train_data = []
    train_data = train_vectors
    print(train_data.shape)
    print('', time.time()-prev, 's')
    prev = time.time()

    def swap(vec):
        return np.concatenate([vec[:4], vec[5:6], vec[4:5]], axis=0)
    def iterate(dataset, vectors):
        vecL = [vectors[i] if v["answer"] == u"1" else swap(vectors[i]) for i, (k, v)
                in enumerate(sorted(dataset.iteritems(), key=lambda x:x[0]))]
        matrix = np.concatenate(vecL, axis=0)
        return matrix.reshape((len(vecL), matrix.shape[0]/len(vecL), matrix.shape[1]))

    print(' set vectors valid, test')
    valid_data = iterate(valid_data, valid_vectors)
    print(valid_data.shape)
    test_data = iterate(test_data, test_vectors)
    print(test_data.shape)
    print('', time.time()-prev, 's')
    prev = time.time()

    print('# of train data =', len(train_data))
    print('# of valid data =', len(valid_data))
    print('# of test data =', len(test_data))

    return train_data, valid_data, test_data


def setup_model(args):
    if args.model_type.lower() == "gru":
        model = GRU(args)
    elif args.model_type.lower() == "grumulti":
        model = GRUmulti(args)
    elif args.model_type.lower() == "grumultimax":
        model = GRUmultiMax(args)
    elif args.model_type.lower() == "grumultisoftmax":
        model = GRUmultiSoftmax(args)
    elif args.model_type.lower() == "lstm":
        model = LSTM(args)
    elif args.model_type.lower() == "lstmmulti":
        model = LSTMmulti(args)
    elif args.model_type.lower() == "lstmmultimax":
        model = LSTMmultiMax(args)
    elif args.model_type.lower() == "lstmmultisoftmax":
        model = LSTMmultiSoftmax(args)
    elif args.model_type.lower() == "lstmmultiover":
        model = LSTMmultiOver(args)
    elif args.model_type.lower() == "sumcontext":
        model = SumContextBilinear(args)
    elif args.model_type.lower() == "skipcnn":
        model = SkipCNN(args)
    elif args.model_type.lower() == "baseline":
        model = Baseline(args)
    elif args.model_type.lower() == "transfer":
        model = Transfer(args)
    elif args.model_type.lower() == "asymmetric":
        model = Asymmetric(args)
    elif args.model_type.lower() == "asymmetrictanh":
        model = AsymmetricTanh(args)
    elif args.model_type.lower() == "asymmetrictanhdeep":
        model = AsymmetricTanhDeep(args)
    else:
        print('set valid model type name')
        exit()

    optimizer = model.setup_optimizer()
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()
    return model, optimizer


def train_model(args):
    print('load data')
    train_data, valid_data, test_data = load_processed_dataset(args.load_corpus)

    print('setup model, optimizer')
    # Replicability of random initialization.
    np.random.seed(args.seed)
    model, optimizer = setup_model(args)

    # Random parameter test
    #print('evaluate test')
    #loss_mean, correct = evaluate(test_data, model, args)
    #print('testloss: {:.5f}, accuracy: {:.4f}\t{}'.format(
    #    loss_mean, float(correct/len(test_data)), datetime.today().strftime("%Y/%m/%d,%H:%M:%S") ))

    best_valid = 0.5# start from chance rate
    subbest_test = 0.
    
    for i_epoch in range(args.n_epoch):
        # Training
        train_epoch(model, optimizer, train_data, valid_data, i_epoch, args)

        # Evaluate on validation data
        print('evaluate')
        loss_mean, correct = evaluate(valid_data, model, args)
        print('epoch {} validloss: {:.5f}, accuracy: {:.4f}\t{}'.format(
            i_epoch, loss_mean, float(correct/len(valid_data)),
            datetime.today().strftime("%Y/%m/%d,%H:%M:%S") ))

        # If new record in validation data, update record and test in test data
        # Note: Even if the existing best TEST record is better than the new TEST record,
        #       it must be updated if meeting new VALIDATION record,
        #       because we can only see validation results when training.
        if float(correct/len(valid_data)) > best_valid:
            print('** new record')
            if args.save_path_model:
                save_name = args.save_path_model + args.model_name + \
                            ".%d.%s." % (i_epoch+1, datetime.today().strftime("%Y%m%d.%H%M%S"))
                save(model, optimizer, save_name, args)
            best_valid = float(correct/len(valid_data))

            print('evaluate test')
            loss_mean, correct = evaluate(test_data, model, args)

            print('epoch {} testloss: {:.4f}, accuracy: {:.3f}\t{}'.format(
                i_epoch, loss_mean, float(correct/len(test_data)),
                datetime.today().strftime("%Y/%m/%d,%H:%M:%S") ))
            subbest_test = float(correct/len(test_data))
            
    # Evaluate on test dataset (no use of early stopping)
    #print('evaluate test')
    #loss_mean, correct = evaluate(test_data, model, args)
    #print('epoch {} testloss: {:.4f}, accuracy: {:.3f}\t{}'.format(
    #    i_epoch, loss_mean, float(correct/len(test_data)),
    #    datetime.today().strftime("%Y/%m/%d,%H:%M:%S") ))

    print('*** best valid result', best_valid)
    print('*** (subbest) test result', subbest_test)


def save(model, optimizer, save_name, args):
    serializers.save_npz(save_name+"model", copy.deepcopy(model).to_cpu())
    serializers.save_npz(save_name+"optimizer", optimizer)
    print('save', save_name)


def make_batch(datas, train=True):
    allconcat = np.concatenate(datas, axis=0)
    if args.gpu >= 0:
        allconcat = cuda.cupy.array(allconcat)
    batch = xp.split(allconcat, allconcat.shape[1], axis=1)
    batch = [xp.reshape(x, (x.shape[0], x.shape[2])) for x in batch]
    return batch


def get_neg(pos, x_batch_seq, negative=-1):
    if negative <= 0:
        neg = xp.concatenate([pos[1:], pos[0:1]], axis=0)
    else:
        switch = np.random.randint(0, negative)# stochastically switch negative type
        # Note: np.random.randint can return 0,1,2,...,`negative-1`. (not including `negative`.)
        if switch <= 3:# rewind negative
            neg = x_batch_seq[switch]
            # Note: If you use one-by-one learning, this negative example can be not negative example.
            # (e.g., when this is the 3rd event and one-by-one learner
            #  has to predict anything of 2-5th or 3-5th events.)
        else:# assign random other positive 5th event
            neg = xp.concatenate([pos[1:], pos[0:1]], axis=0)
            # Just make 1-shifted batch (rotation) from positive 5-th event batch,
            # that is, the batch-idx 1's positive becomes the batch-dix 2's negative
            # and the batch-idx N's positive becomes the batch-dix 1's negative.
            # (Mini-batch sampling is randomly done. So exploit the randomness in this time.)
    return neg


def train_epoch(model, optimizer, train_data, valid_data, epoch, args):
    model.zerograds()
    whole_len = len(train_data)
    sum_loss_data = xp.zeros(())
    
    # Replicability of mini-batch samplings.
    np.random.seed(args.seed + epoch)
    perm = np.random.permutation(len(train_data)).tolist()

    cur_at = time.time()
    sum_correct, processed, prev_i = 0., 0, 0
    print("Epoch",epoch,"start.")

    for i in six.moves.range(0, len(train_data), args.batchsize):
        model.zerograds()

        # Make batch
        batch_ids = perm[i:i+args.batchsize]
        processed += len(batch_ids)
        x_batch_seq = make_batch([train_data[idx:idx+1] for idx in batch_ids])

        # Calculate
        x_batch_seq, pos = x_batch_seq[:4], x_batch_seq[4]
        neg = get_neg(pos, x_batch_seq, args.negative_type)

        # Replicability of dropout. All seeds should not be overlapped in whole training step.
        model.xp.random.seed(args.seed + epoch * len(train_data) / args.batchsize + i)
        # Calculate loss
        loss, correct = model.solve(x_batch_seq, pos, neg,
                                    train=True, variablize=True, onebyone=args.onebyone)

        sum_loss_data += loss.data
        sum_correct += correct

        # Backward, Update
        loss.backward()
        optimizer.update()

        # Print loss and acc
        if processed >= args.per:
            loss_mean = cuda.to_cpu(sum_loss_data) / processed
            now = time.time()
            throuput = processed * 1.0 / (now - cur_at)

            print('epoch {} iter {} loss: {:.5f}, accuracy: {:.4f} ({:.2f} iters/sec)\t{}'.format(
                epoch, (i + len(batch_ids)), loss_mean,
                float(sum_correct/processed), throuput,
                datetime.today().strftime("%Y/%m/%d,%H:%M:%S") ))
            cur_at = now
            sum_loss_data.fill(0)
            sum_correct, processed = 0., 0

        # Evaluate on validation set on the way
        if i-prev_i > whole_len / args.frac_eval:
            prev_i = i
            now = time.time()
            print('evaluate')
            loss_mean, correct = evaluate(valid_data, model, args)
            print('epoch {} iter {} validloss: {:.5f}, accuracy: {:.4f} ({:.2f} iters/sec)\t{}'.format(
                epoch, (i + len(batch_ids)), loss_mean,
                float(correct/len(valid_data)), throuput,
                datetime.today().strftime("%Y/%m/%d,%H:%M:%S") ))
            cur_at += time.time() - now  # skip time of evaluation
            if args.save_path_model:
                save_name = args.save_path_model + args.model_name + \
                            ".%d.%s." % (epoch+(i*1.0/whole_len), datetime.today().strftime("%Y%m%d.%H%M%S"))
                save(model, optimizer, save_name, args)

def evaluate(dataset, model, args):
    sum_correct = 0.
    sum_loss_data = xp.zeros(())
    for i in six.moves.range(0, len(dataset), args.batchsize):
        x_batch_seq = make_batch([dataset[i+j:i+j+1] for j in range(args.batchsize)], train=False)
        x_batch_seq, pos, neg = x_batch_seq[:4], x_batch_seq[4], x_batch_seq[5]
        loss, correct = model.solve(x_batch_seq, pos, neg, train=False, variablize=True)
        sum_loss_data += loss.data
        sum_correct += correct
    return cuda.to_cpu(sum_loss_data) / len(dataset), sum_correct


if __name__ == "__main__":
    print("##### ##### ##### #####")
    print(" ".join(sys.argv))
    print("STARTING TIME:",datetime.today().strftime("%Y/%m/%d %H:%M:%S"))
    print("##### ##### ##### #####")
    for k, v in sorted(args.__dict__.items(), key=lambda x:len(x[0])): print("#",k,":\t",v)
    print("##### ##### ##### #####")

    # Training
    train_model(args)

    print(' ***** E N D ***** ')
