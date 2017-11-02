#!/usr/bin/env python
import optparse
import sys
import models
from collections import namedtuple
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

sys.stderr.write("Decoding %s...\n" % (opts.input,))



for f in french:

  # distortion limit
  if len(f) > 5:
    dist_limit = 5  
  else:
    dist_limit = len(f)
  # # computing future cost table
  fcost = [[0] for _ in range(len(f))]
  for i in range(len(f)):
    fcost[i].append(tm1[f[i:i+1]][0].logprob)
    for j in range(i+1, len(f)):
      fcost[i].append(fcost[i][-1]  + tm[f[j:j+1]][0].logprob)
      for k in range(i,j):
        if f[k:j+1] in tm1:
          if fcost[i][k-i] + tm1[f[k:j+1]][0].logprob > fcost[i][j-i+1]:
            fcost[i][j-i+1] = fcost[i][k-i] + tm1[f[k:j+1]][0].logprob
    fcost[i].remove(fcost[i][0])
    fcost.append([0])
  del i,j

  hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase, cover, cover_end")
  initial_hypothesis = hypothesis(fcost[0][-1], lm.begin(), None, None, [0 for _ in range(len(f))], 0)
  queue = {lm.begin():initial_hypothesis}

  while True:
    print len(queue)

    top = max(queue.itervalues(), key=lambda h: h.logprob)
    
    if 0 not in top.cover:
      winner = top
      break;
    else:

      # for j in xrange(i + 1, len(f) + 1):
      for j in range(len(f)):
        for k in range(j + 1, len(f)+1):
          if 1 not in top.cover[j:k]:
            if f[j:k] in tm:
              for phrase in tm[f[j:k]]:
                
                logprob = top.logprob + phrase.logprob
                lm_state = top.lm_state

                for word in phrase.english.split():

                  (lm_state, word_logprob) = lm.score(lm_state, word)
                  logprob += word_logprob

                cover = top.cover[:]
                cover[j:k] = [1 for _ in range(k-j)]
                pdb.set_trace()
                # if k > top.cover_end:
                  
                #   logprob += fcost[k][-1] - fcost[top.cover_end][-1]
                #   cover_end = k - 1

                m = j; n = k - 1;
                while m >= 0 and top.cover[m] == 0:
                  m -= 1
                while n <= len(f) - 1 and top.cover[n] == 0:
                  n += 1
                if m != j - 1:
                  logprob += fcost[m+1][j-m-2]
                if n != k:
                  logprob += fcost[k][n-k-1]

                logprob -= fcost[m+1][n-m-2]

                #   logprob += fcost[k][-1]# - fcost[j][-1] ######################coarse
                logprob += lm.end(lm_state) if k == len(f) else 0.0#fcost[j][-1]

                new_hypothesis = hypothesis(logprob, lm_state, top, phrase, cover, top.cover_end)
                if lm_state not in queue or queue[lm_state].logprob < logprob: # second case is recombination
                  queue[lm_state] = new_hypothesis

      del queue[top.lm_state]

      queue_prune = {}
      for h in sorted(queue.itervalues(),key=lambda h: -h.logprob)[:opts.s]:
        queue_prune[h.lm_state] = h
      queue = queue_prune

  def extract_english(h): 
    return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)
  print extract_english(winner)
  pdb.set_trace()

  if opts.verbose:
    def extract_tm_logprob(h):
      return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
    tm_logprob = extract_tm_logprob(winner)
    sys.stderr.write("LM = %f, TM = %f, Total = %f\n" % 
      (winner.logprob - tm_logprob, tm_logprob, winner.logprob))
