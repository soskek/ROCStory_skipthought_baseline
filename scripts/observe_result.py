#!/usr/bin/env python
# coding:utf-8
import argparse
import readline  # flake8: noqa
import sys
#import sklearn
#import sklearn.decomposition
import numpy
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.font_manager import FontProperties
"""
epoch 28 validloss: 0.87789, accuracy: 0.6595   2016/04/19,12:07:10
epoch 21 testloss: 0.8971, accuracy: 0.656      2016/04/19,12:03:35
epoch 27 iter 8192 loss: 0.13881, accuracy: 0.9465 (1082.86 iters/sec)  2016/04/19,12:05:58
"""
#iter 175 training perp: 1.52, accuracy: 0.840 (difficulty:23.61) (1.71 iters/sec)
#maxtr = 0
#maxts = 0

over_lim = 1.0

#mycm = cm.rainbow
#mycm = cm.nipy_spectral
mycm = cm.gist_rainbow
#mycm = cm.gist_ncar
#mycm = cm.hsv
#mycm = cm.brg
#mycm = cm.spectral
time = False
#time = True
mode = "acc"
f_s = sys.argv[1:]
if f_s[-1].replace(".","",1).isdigit():
    over_lim = float(f_s[-1])
    print "over_lim is", over_lim
    f_s = f_s[:-1]
if f_s[-1] in ["perp","loss"]:
    mode = "perp"
    print "mode is", "perp"
    f_s = f_s[:-1]
if f_s[-1].replace(".","",1).isdigit():
    over_lim = float(f_s[-1])
    print "over_lim is", over_lim
    f_s = f_s[:-1]
f_s = ["./"+w_ if "/" not in w_ else w_ for w_ in f_s]
shared_dir = "/".join(f_s[0].split("/")[:-1])+"/"
n_file = len(f_s)

c_ = 0
for f_name in f_s:
    if f_name.startswith(shared_dir):
        continue
    else:
        while shared_dir:
            shared_dir = "/".join(shared_dir.split("/")[:-2])+"/"
            if f_name.startswith(shared_dir):
                break
            if not shared_dir:
                shared_dir = "./"
            c_ += 1
            if c_ > 100:
                shared_dir = "/home/sosuke.k/"
                break

print shared_dir

if mode == "perp":
    maxtr = 10000
    maxts = 10000
else:
    maxtr = 0
    maxts = 0


max_it = 0

for kth, f_name in enumerate(f_s, start=1):
    D = {}
    vD = {}
    aD = {}
    avD = {}
    L = []
    ep = 0
    it = 0
    loop = 0
    jump = 45502
    input_lines = []

    test_avD = {}
    test_vD = {}
    ###############

    for line in open(f_name):
        if "Epoch" in line:
            ep = float(line.split()[1])
        if "loss" in line and line.startswith("epoch"):
            sp = line.strip().split()
            ep = float(sp[1])
            if "validloss" in line:
                avD[ep*jump] = float(sp[5])
                vD[ep*jump] = float(sp[3][:-1])
            elif "testloss" in line:
                test_avD[ep*jump] = float(sp[5])
                test_vD[ep*jump] = float(sp[3][:-1])
            else:
                it = float(sp[3]) + jump*int(ep)
                aD[it] = float(sp[7])
                D[it] = float(sp[5][:-1])
        if "*** (subbest) test result" in line:
            test_best = float(line.strip().split()[-1])

    #*** (subbest) test result 0.655799037948
    """
    epoch 28 validloss: 0.87789, accuracy: 0.6595   2016/04/19,12:07:10
    epoch 21 testloss: 0.8971, accuracy: 0.656      2016/04/19,12:03:35
    epoch 27 iter 8192 loss: 0.13881, accuracy: 0.9465 (1082.86 iters/sec)  2016/04/19,12:05:58
    """

    f_name = f_name.replace(shared_dir,"")
    max_it = max(it, max_it)

    ####################
    for ep, a in sorted(D.items(), key=lambda x:x[0]):
        L.append( (ep, a) )
    vL = []
    for ep, a in sorted(vD.items(), key=lambda x:x[0]):
        vL.append( (ep, a) )

    ###############
    aL = []
    avL = []
    for ep, a in sorted(aD.items(), key=lambda x:x[0]):
        aL.append( (ep, a) )
    for ep, a in sorted(avD.items(), key=lambda x:x[0]):
        avL.append( (ep, a) )

    ################

    if mode == "perp":
        BEST = min
    else:
        BEST = max
        L = aL
        vL = avL
    train = 1
    if not L:
        print "-",f_name
        continue
    b = 0
    x_, y_ = zip(*L)
    if train:
        v = np.array(y_)
        x_, y_ = zip(*[(L[j][0], max(v[j:j+b+1])) for j in xrange(len(L)-b)])
        plt.plot( x_, y_, dashes=(1.5,1), alpha=0.6, label=f_name+" TRAIN", lw=0.3, color=mycm(float(kth) / n_file))
#    maxtr = max(maxtr, max(y_))
    maxtr = BEST(maxtr, BEST(y_))

    if len(vL) >= 2:
        x, y = zip(*vL)
        b = min([2, max([len(vL)-2, 0])])
        v = np.array(y)
        x, y = zip(*[(vL[j][0], max(v[j:j+b+1])) for j in xrange(len(vL)-b)])
#        plt.plot( x, y, alpha=0.9, label=f_name+" VALID", lw=0.4, color=mycm(float(kth) / n_file))
        plt.plot( x, y, "_-", ms=4, alpha=0.9, label=f_name+" VALID", lw=0.4, color=mycm(float(kth) / n_file))
        plt.axhline(y=BEST(y), color=mycm(float(kth) / n_file), lw=0.1, alpha=0.5)
        x, y = zip(*vL)
        #    maxts = max(maxts, max(y))
        maxts = BEST(maxts, BEST(y))
#    plt.ylim(0.97,1.0)
#    plt.legend(loc='lower center', prop={'size' : 10}, ncol=3)
    if len(f_name) > 33:
        ncol = 1
    elif len(f_name) > 22:
        ncol = 2
    else:
        ncol = 3
    plt.legend(loc='best', prop={'size' : 4}, ncol=ncol)
    if time:
        print "@",f_name, "\t\t%.1lf seconds" % max(x_)
    else:
        print "@",f_name, "\t\t%d epoch" % int(max(x_)/jump), "\t%d iter" % max(x_), 
    if len(vL) >= 2:
        for tu in reversed(sorted(avD.items(), key=lambda tu:tu[1])):
            if tu[0] in test_avD:
                test_score = test_avD[ tu[0] ]
                break

        print "\tTEST: %.3lf" % test_score, "\tVALID BEST: %.3lf" % BEST(y), "\tTRAIN BEST: %.3lf" % BEST(y_)
        print "\tTRAIN Last seq:", " ".join(["%.3lf" % v for v in y_[-10:]])
        print "\tVALID Last seq:\t\t  ", " ".join(["%.3lf" % v for v in y[-10:]])
    else:
        if L:
            print "\t\t", "\tTRAIN BEST: %.3lf" % BEST(y_)
            print "\tTRAIN Last seq:", " ".join(["%.3lf" % v for v in y_[-10:]])


#plt.yticks( np.arange(1000)/10000.0 )

if mode == "perp":
#    plt.yticks(numpy.arange(0,15.0,0.2), fontsize=4)
    plt.yticks(numpy.arange(0,2.,0.1), fontsize=4)
#    plt.ylim(1.0,over_lim)
    plt.ylim(0.05,over_lim)
    for tmp_ in numpy.arange(0,2.,0.05):
        plt.axhline(y=tmp_, ls="--", color='gray', lw=0.1, alpha=0.3)
else:
    plt.yticks(numpy.arange(0,1.0,0.05), fontsize=4)
    plt.ylim(0.55,over_lim)
    for tmp_ in numpy.arange(0,1.0,0.01):
        plt.axhline(y=tmp_, ls="--", color='gray', lw=0.1, alpha=0.3)

epochs = numpy.arange(0, max_it+40000,jump)
plt.xlim(0, max_it+40000)
#plt.xticks(epochs, fontsize=8)
for e_ in epochs:
    plt.axvline(x=e_, color='blue', lw=0.2, alpha=0.3)

plt.axhline(y=maxtr, color='gray', lw=0.2, alpha=0.3)
plt.axhline(y=maxts, color='black', lw=0.2, alpha=0.5)

#plt.axhline(y=0.618, color='red', lw=0.1, alpha=0.2)
#plt.axhline(y=0.634, color='red', lw=0.1, alpha=0.4)
#plt.axhline(y=0.662, color='red', lw=0.2, alpha=0.6)

plt.savefig('output.png', bbox_inches='tight', dpi=300)
plt.close()
