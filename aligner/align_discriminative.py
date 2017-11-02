#!/usr/bin/env python
import optparse
import sys
import numpy as np
from collections import defaultdict

optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.2, type="float", help="Threshold for aligning with Dice's coefficient (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)

sys.stderr.write("Training begin...")
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]

# for n,s in enumerate(bitext):
	# s[0],s[-1] = s[-1], s[0]

#gibbs start

#initial alignment
alignment = defaultdict(int)
for (n, (f, e)) in enumerate(bitext):
	seq = []
	for i in e:
		seq.append(np.random.randint(0,len(f)))
	alignment[n] = seq

#initial table
table = defaultdict(int)
for key in alignment:
	for n,pos in enumerate(alignment[key]):
		#table format:(f,e)
		table[bitext[key][0][pos],bitext[key][1][n]] += 1

for k in range(1000):
	
	for (n, (f, e)) in enumerate(bitext):
		for j,e_j in enumerate(e):
			#initial:
			numerator = defaultdict(int)
			denominator = 0
			prob = []
			
			#do gibbs for e_j
			f_pos = alignment[n][j]
			aligned = f[f_pos]
			table[aligned,e_j] -= 1
			
			for key in table:
				if key[1] == e_j:
					for f_i in set(f):
						if key[0] == f_i:
							numerator[key[0]] += table[key] + 10**-8
			#calculate probability
			summation = sum(numerator.values())
			for key, val in numerator.items():
				numerator[key] = float(val)/ summation 
			
			#random choose
			array = [i for i in range(len(numerator.keys()))]
			chosen = np.random.choice(array,p=numerator.values())
			f_i = numerator.keys()[chosen]
			#update count table and alignment
			table[f_i,e_j] += 1
			alignment[n][j] = f.index(f_i)
	sys.stderr.write("%i iter" % k)
	sys.stderr.write("\n")
			
#output
for key in alignment:
	for n, align in enumerate(alignment[key]):
		sys.stdout.write("%i-%i " % (n,align))
	sys.stdout.write("\n")
