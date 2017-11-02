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
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=10000000, type="int", help="Number of sentences to use for training and alignment")
optparser.add_option("-s", "--save", dest="save", default='./parameters', type="str", help="place to save parameters")
(opts, _) = optparser.parse_args()

f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)

sys.stderr.write("Training hmm ...")
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]
f_count = defaultdict(int)
e_count = defaultdict(int)
fe_count = defaultdict(int)


# transition prob
p_trans = defaultdict(float)
p_trans_sum = defaultdict(float)
# calculate maximum length
max_len_e = 0
max_len_f = 0
lengths_e = set()
for (n, (f, e)) in enumerate(bitext):
  len_e = len(e)
  max_len_e = max(max_len_e, len(e))
  max_len_f = max(max_len_f, len(f))
  lengths_e.add(len(e))
  for word_e in e:
    e_count[word_e] += 1
    for word_f in f:
      fe_count[(word_f, word_e)] += 1
  for word_f in f:
    f_count[word_f] += 1

  for i in range(len_e):
    for i_prime in range(len_e):
      p_trans[(i, i_prime, len_e)] += 1
      p_trans_sum[(i_prime, len_e)] += 1

sys.stderr.write("\n")

'''
initialization
'''
for len_e in lengths_e:
  for i in range(len_e):
    for i_prime in range(len_e):
      p_trans[(i, i_prime, len_e)] /= p_trans_sum[(i_prime, len_e)]

# init prob
p_init = [1 / max_len_e for i in range(max_len_e)]

# word prob
p_fe = defaultdict(float)
for word_f, word_e in fe_count.keys():
    p_fe[(word_f, word_e)] = fe_count[(word_f, word_e)] / e_count[word_e]


'''
training:
'''
start_time = time.time()
MaxEpoch = 20
for epoch_idx in range(MaxEpoch):
  sys.stderr.write("Start Training {} / {}, {} s passed\n".format(epoch_idx + 1, MaxEpoch, time.time() - start_time))
  '''
  initial counts
  '''
  c_trans = defaultdict(float)
  c_fe = defaultdict(float)
  c_d = defaultdict(float)
  c_init = [random.random() for i in range(max_len_e)]

  for (n, (f, e)) in enumerate(bitext):
    len_e = len(e)
    len_f = len(f)

    # normalized inital probability
    p_init_sum = sum(p_init[:len_e])    
    p_init_norm = [item / p_init_sum for item in p_init[:len_e]]
    # p_init_norm = p_init

    Q = defaultdict(float)

    '''
    alpha
    forward
    '''

    # base
    alpha = [[0 for j in range(len_f)] for i in range(len_e)]
    for i in range(len_e):
      alpha[i][0] = p_init_norm[i] * p_fe[(f[0], e[i])]
      Q[0] += alpha[i][0]

    for i in range(len_e):
      alpha[i][0] /= Q[0]

    # recursive
    for j in range(1, len_f):
      for i in range(len_e):
        alpha[i][j] = \
          p_fe[(f[j],e[i])] * sum([alpha[i_prime][j-1] * p_trans[(i, i_prime, len_e)] for i_prime in range(len_e)])
        Q[j] += alpha[i][j]
      for i in range(len_e):
        alpha[i][j] /= Q[j]
    

    '''
    beta
    backward
    '''
    
    # base
    beta = [[0 for j in range(len_f)] for i in range(len_e)]
    for i in range(len_e):
      beta[i][len_f - 1] = 1.0 / Q[len_f - 1]

    # recursive
    for j in reversed(range(1, len_f)):
      for i_prime in range(len_e):
        beta[i_prime][j - 1] = 1.0 / Q[j - 1] * sum([beta[i][j] * p_trans[(i, i_prime, len_e)] * p_fe[(f[j], e[i])] for i in range(len_e)])
    
    '''
    gamma
    posterior
    '''
     # word count
    gamma_sum = sum([alpha[i][len_f - 1] for i in range(len_e)])
    for i in range(len_e):
      c_init[i] += alpha[i][0] * beta[i][0] * Q[0] / gamma_sum
      for j in range(len_f):
        gamma_i_j = alpha[i][j] * beta[i][j] * Q[j] / gamma_sum
        c_fe[(f[j], e[i])] += gamma_i_j
        c_fe[(e[i])] += gamma_i_j
   
    
    '''
    ksi
    joint prob
    '''

    for i in range(len_e):
      for i_prime in range(len_e):
        for j in range(1, len_f):
          c_d[i - i_prime] += alpha[i_prime][j - 1] * p_trans[(i, i_prime, len_e)] * beta[i][j] * p_fe[(f[j],e[i])] / gamma_sum

    
    '''
    alignment count
    count
    '''

    for i in range(len_e):
      for i_prime in range(len_e):
        c_trans[(i_prime, i, len_e)] += c_d[i - i_prime]
        c_trans[(i_prime, len_e)] += c_d[i - i_prime]
        
  
  '''
  updates
  '''
  # update parameters
  for word_f, word_e in p_fe.keys(): 
    p_fe[(word_f, word_e)] = c_fe[(word_f, word_e)] / c_fe[(word_e)]
  
  for i, i_prime, len_e in p_trans.keys():
    p_trans[(i, i_prime, len_e)] = c_trans[(i_prime, i, len_e)] /  c_trans[(i_prime, len_e)]
    
  sum_init = sum(c_init)
  p_init = [cc / sum_init for cc in c_init]


  '''
  Save parameters
  '''
  with open(opts.save+'.'+str(epoch_idx), 'w+') as f:
    pickle.dump([p_init, p_fe, p_trans], f)
  '''

decoding
viterbi search
'''
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

  result.append(' '.join(['{}-{}'.format(j, i) for i, j in reversed(path)]) + '\n')

  with open('./hmm.out', 'w+') as f:
    for r in result:
      f.write(r)
