#!/usr/bin/env python
import optparse
import time
import sys
import models
from collections import namedtuple
import numpy as np
import pdb
optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=10, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=10000, type="int", help="Maximum stack size (default=1)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
opts = optparser.parse_args()[0]

tm = models.TM(opts.tm, opts.k)
tm1 = models.TM(opts.tm, opts.k)  # for computing future cost
lm = models.LM(opts.lm)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

# tm should translate unknown words as-is with probability 1
for word in set(sum(french,())):
    if (word,) not in tm:
        tm[(word,)] = [models.phrase(word, 0.0)]
        tm1[(word,)] = [models.phrase(word, 0.0)]

def extract_english(h): 
    return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)

def lm_future_cost(phrase):
  lm_score = 0
  lm_state = lm.begin()
  if phrase in tm1:
    english_phrase = tm1[phrase][0].english.split()
    for word in english_phrase:
      (lm_state, word_logprob) = lm.score(lm_state, word)
      lm_score += word_logprob
  else:
    lm_score = -9999 
  
  return lm_score

sys.stderr.write("Decoding %s...\n" % (opts.input,))

for f in french:

  # # computing future cost table
  fcost = [[0] for _ in range(len(f))]
  for i in range(len(f)):
    fcost[i].append(tm1[f[i:i+1]][0].logprob)
    for j in range(i+1, len(f)):
      fcost[i].append(fcost[i][-1]  + tm1[f[j:j+1]][0].logprob + lm_score)
      for k in range(i,j):
        if f[k:j+1] in tm1:
          if fcost[i][k-i] + tm1[f[k:j+1]][0].logprob > fcost[i][j-i+1]:
            fcost[i][j-i+1] = fcost[i][k-i] + tm1[f[k:j+1]][0].logprob
    fcost[i].remove(fcost[i][0])
    fcost.append([0])

  del i,j

  hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase, coverage")
  initial_hypothesis = hypothesis(fcost[0][-1], lm.begin(), None, None, np.zeros(len(f)))
  stacks = [{} for _ in range(len(f))] + [{}]
  # inital stack
  stacks[0][lm.begin()] = initial_hypothesis
  for i, stack in enumerate(stacks[:-1]):
      for h in sorted(stack.itervalues(),key=lambda h: -h.logprob)[:opts.s]: # prune
          # new hyp
          for f_s in xrange(len(f)):
              for f_e in xrange(f_s + 1, len(f) + 1):
                  if 1 not in h.coverage[f_s:f_e]:
                      if f[f_s:f_e] in tm:
                          for phrase in tm[f[f_s:f_e]]:
                              # previous score and current tm score
                              logprob = h.logprob + phrase.logprob
                              # previous words
                              lm_state = h.lm_state
                              
                              # current language model score
                              for word in phrase.english.split():
                                  (lm_state, word_logprob) = lm.score(lm_state, word)
                                  logprob +=  word_logprob
                              
                              coverage = np.copy(h.coverage)
                              coverage[f_s:f_e] = 1
                              # cover_end = f_e-1 if f_e > cover_end
                              
                              # if f_e > h.cover_end:
                              #   cover_end = f_e - 1
                              #   logprob += fcost[f_e][-1] - fcost[h.cover_end][-1]
                              m = f_s; n = f_e - 1;
                              while m >= 0 and h.coverage[m] == 0:
                                m -= 1
                              while n <= len(f) - 1 and h.coverage[n] == 0:
                                n += 1
                              if m != f_s - 1:
                                logprob += fcost[m+1][f_s-m-2]
                              if n != f_e:
                                logprob += fcost[f_e][n-f_e-1]

                              logprob -= fcost[m+1][n-m-2]


                              logprob += lm.end(lm_state) if f_e == len(f) else 0.0
                              new_hypothesis = hypothesis(logprob, lm_state, h, phrase, coverage)
                              
                              target = i + f_e - f_s

                              if lm_state not in stacks[target] or stacks[target][lm_state].logprob < logprob:
                                  stacks[target][lm_state] = new_hypothesis
                                      #time.sleep(2)
        #for value in sorted(stack.itervalues(),key=lambda h: -h.logprob)[:opts.s]:
        #    print value.coverage
        #    print extract_english(value),value.logprob
        #raw_input('===================')
  winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)
  print extract_english(winner)
  # pdb.set_trace()

  if opts.verbose:
      def extract_tm_logprob(h):
          return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
      tm_logprob = extract_tm_logprob(winner)
      sys.stderr.write("LM = %f, TM = %f, Total = %f\n" % 
              (winner.logprob - tm_logprob, tm_logprob, winner.logprob))
