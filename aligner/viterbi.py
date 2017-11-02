#!/usr/bin/env python
'''
hmm alginment
'''
from __future__ import division
from __future__ import print_function
import optparse
import sys
from collections import defaultdict
import random
import math
import time
import pickle 
optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="Threshold for aligning with Dice's coefficient (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=1000, type="int", help="Number of sentences to use for training and alignment")
optparser.add_option("-s", "--save", dest="save", default='./parameters', type="str", help="place to save parameters")
optparser.add_option("-m", "--model", dest="model", type="str", help="parameters to be tested")
(opts, _) = optparser.parse_args()

f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)

bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]
sys.stderr.write('Reading parameters...\n')
with open(opts.model, 'r+') as f:
     p_init, p_fe, p_trans = pickle.load(f)
sys.stderr.write('Decoding...\n')
result =[]
for (n, (f, e)) in enumerate(bitext):
  len_e = len(e)
  len_f = len(f)
  V = [[0 for j in range(len(f))] for i in range(len(e))]
  
  # base
  for i in range(len_e):
    V[i][0] = [-1, -1, math.log(1e-12 + p_init[i]) +  math.log(p_fe[(f[0], e[i])] + 1e-12)]
  
  for j in range(1, len_f):
    for i in range(len_e):
      max_score = -float('inf')
      max_i = None
      max_jm1 = None
      for i_prime in range(len_e):
        curr_score = V[i_prime][j-1][2] + math.log(1e-12 + p_trans[(i, i_prime, len_e)]) + math.log(p_fe[(f[j],e[i])] + 1e-12) 
        if curr_score > max_score:
          max_score = curr_score
          max_i, max_jm1 = i_prime, j - 1
      V[i][j] = [max_i, max_jm1, max_score]
  
  max_score = -float('inf')
  max_last_i = len_e - 1
  
  for i in range(len_e):
    if V[i][len_f - 1][2] > max_score:
      max_score = V[i][len_f - 1][2]
      max_last_i = i

  path = []
  max_j = len_f - 1
  for j in reversed(range(len_f)):
    path.append([max_last_i, j])
    max_last_i= V[max_last_i][j][0]

  result.append(' '.join(['{}-{}'.format(j, i) for i, j in reversed(path)]))

for string in result:
    print(string)
'''
  with open('./hmm.out', 'w+') as f:
    for r in result:
      f.write(r)
      '''