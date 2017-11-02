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

  hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase, cover_end")
  initial_hypothesis = hypothesis(fcost[0][-1], lm.begin(), None, None, 0)
  queue = {lm.begin():initial_hypothesis}

  while True:
    # print len(queue)
    top = max(queue.itervalues(), key=lambda h: h.logprob)
    
    if top.cover_end == len(f):
      winner = top
      break;
    else:
      i = top.cover_end

      for j in xrange(i , len(f)):
        if f[i:j+1] in tm:
          for phrase in tm[f[i:j+1]]:
            logprob = top.logprob + phrase.logprob
            lm_state = top.lm_state

            for word in phrase.english.split():

              (lm_state, word_logprob) = lm.score(lm_state, word)
              logprob += word_logprob

            logprob += lm.end(lm_state) if j + 1 == len(f) else fcost[j+1][-1] - fcost[i][-1]

            new_hypothesis = hypothesis(logprob, lm_state, top, phrase, j+1)
            if lm_state not in queue or queue[lm_state].logprob < logprob: # second case is recombination
              queue[lm_state] = new_hypothesis

      if i < len(f)-1:
        if (f[i+1],) in tm and (f[i],) in tm:
          for word1 in tm[(f[i+1],)]:
            logprob = top.logprob + word1.logprob
            lm_state = top.lm_state

            (lm_state, word_logprob) = lm.score(lm_state, word1.english)
            logprob += word_logprob

            for word2 in tm[(f[i],)]:
              logprob_temp = logprob + word2.logprob
              (lm_state_temp, word_logprob_temp) = lm.score(lm_state, word2.english)

              logprob_temp += word_logprob_temp
              logprob_temp += lm.end(lm_state_temp) if i+2 == len(f) else fcost[i+2][-1] - fcost[i][-1]

              phrase = word2
              phrase = phrase._replace(english = word1.english + ' ' + word2.english)
              phrase = phrase._replace(logprob = word1.logprob + word2.logprob)

              new_hypothesis2 = hypothesis(logprob_temp, lm_state_temp, top, phrase, i+2)

              if lm_state_temp not in queue or queue[lm_state_temp].logprob < logprob_temp: # second case is recombination
                queue[lm_state_temp] = new_hypothesis2

      del queue[top.lm_state]

      queue_prune = {}
      for h in sorted(queue.itervalues(),key=lambda h: -h.logprob)[:opts.s]:
        queue_prune[h.lm_state] = h
      queue = queue_prune

  def extract_english(h): 
    return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)
  print extract_english(winner)
  # pdb.set_trace()

  if opts.verbose:
    def extract_tm_logprob(h):
      return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
    tm_logprob = extract_tm_logprob(winner)
    sys.stderr.write("LM = %f, TM = %f, Total = %f\n" % 
      (winner.logprob - tm_logprob, tm_logprob, winner.logprob))
