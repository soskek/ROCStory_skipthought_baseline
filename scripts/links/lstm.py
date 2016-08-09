#!/usr/bin/env python
from __future__ import print_function
import sys

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers
from chainer import optimizers
import numpy as np


class LSTM(chainer.Chain):
    def __init__(self, args):
        super(LSTM, self).__init__(
            # RNN
            LSTM=L.LSTM(args.n_in_units, args.n_units),
            #W_predict=L.Linear(args.n_units, args.n_units),
            W_candidate=L.Linear(args.n_in_units, args.n_units),
        )

        #self.act1 = F.tanh
        self.act1 = F.identity

        self.args = args
        self.n_in_units = args.n_in_units
        self.n_units = args.n_units
        self.dropout_ratio = args.d_ratio
        self.margin = args.margin

        self.initialize_parameters()


    def initialize_LSTM(self, LSTM, initializer):
        initializers.init_weight(LSTM.upward.W.data, initializer)
        initializers.init_weight(LSTM.lateral.W.data, initializer)


    def initialize_parameters(self):
        G_init = initializers.GlorotNormal()

        #initializers.init_weight(self.W_predict.W.data, G_init)
        initializers.init_weight(self.W_candidate.W.data, G_init)
        self.initialize_LSTM(self.LSTM, G_init)


    def calculate_score(self, h, pos, neg, pos_score=None, neg_score=None, multipos=False):
        #h_pro = self.act1(self.W_predict(h))
        h_pro = h
        if multipos:
            # If multiple positive vectors are given,
            # max score is picked up. (other ones are not propagated)
            pos_scoreL = [F.batch_matmul(h_pro, pos_one, transa=True) for pos_one in pos]
            pos_score = F.max(F.concat(pos_scoreL, axis=1), axis=1, keepdims=True)
        else:
            pos_score = F.batch_matmul(h_pro, pos, transa=True)
        neg_score = F.batch_matmul(h_pro, neg, transa=True)

        return pos_score, neg_score


    def solve(self, x_seq, pos, neg, train=True, variablize=False, onebyone=True):
        if variablize:# If arguments are just arrays (not variables), make them variables
            x_seq = [chainer.Variable(x, volatile=not train) for x in x_seq]
            x_seq = [F.dropout(x, ratio=self.dropout_ratio, train=train) for x in x_seq]
            pos = self.act1(self.W_candidate(
                F.dropout(chainer.Variable(pos, volatile=not train),
                          ratio=self.dropout_ratio, train=train)))
            neg = self.act1(self.W_candidate(
                F.dropout(chainer.Variable(neg, volatile=not train),
                          ratio=self.dropout_ratio, train=train)))
        if onebyone and train:
            target_x_seq = [self.act1(self.W_candidate(x)) for x in x_seq[:4]]# 1,2,3,4,5-th targets
            onebyone_loss = 0.

        self.LSTM.reset_state()
        for i, x in enumerate(x_seq):
            h = self.LSTM( F.dropout(x, ratio=self.dropout_ratio, train=train) )
            if onebyone and train and target_x_seq[i+1:]:
                pos_score, neg_score = self.calculate_score(h, target_x_seq[i+1:], neg,
                                                            multipos=True)
                onebyone_loss += F.relu( self.margin - pos_score + neg_score )

        pos_score, neg_score = self.calculate_score(h, pos, neg)
        accum_loss = F.relu( self.margin - pos_score + neg_score )
        TorFs = sum(accum_loss.data < self.margin)
        
        if onebyone and train:
            return F.sum(accum_loss) + F.sum(onebyone_loss), TorFs
        else:
            return F.sum(accum_loss), TorFs

    
    def setup_optimizer(self):
        optimizer = optimizers.RMSpropGraves(
            lr=self.args.start_lr, alpha=0.95, momentum=0.9, eps=1e-08)
        optimizer.setup(self)
        optimizer.add_hook(chainer.optimizer.GradientClipping(self.args.grad_clip))
        optimizer.add_hook(chainer.optimizer.WeightDecay(self.args.weight_decay))
        return optimizer
