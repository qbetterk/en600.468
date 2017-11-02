#!/usr/bin/env python
import optparse
import sys
import os
from collections import defaultdict

optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="Threshold for aligning with Dice's coefficient (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment") #sys.maxint
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)

sys.stderr.write("Training with Dice's coefficient...")
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]

e_count_init = defaultdict(int)
f_count_init = defaultdict(int)
fe_count_init = defaultdict(int)

for (n, (f, e)) in enumerate(bitext):
	for f_i in set(f):
		f_count_init[f_i] = 0
		for e_j in set(e):
			fe_count_init[(f_i,e_j)] = 0
	for e_j in set(e):
		e_count_init[e_j] = 0
	if n % 500 == 0:
		sys.stderr.write(".")

################################  forward f -> e  ##################################

# initialization
p_fe = defaultdict(int)
for (f_i, e_j) in fe_count_init.keys():
	p_fe[(f_i,e_j)] = 1. 

iter_num = 0

while True:
	f_count = f_count_init.copy()
	fe_count = fe_count_init.copy()
	for (k , (f, e)) in enumerate(bitext):
		for e_j in e:
			z = 0
			for f_i in f:
				z += p_fe[(f_i,e_j)]
			for f_i in f:
				c = p_fe[(f_i,e_j)] / z
				fe_count[(f_i,e_j)] += c
				f_count[f_i] += c
		if k % 5000 == 0:
			sys.stderr.write(".")
	sys.stderr.write("\n")

	for (f, e) in fe_count.keys():
		p_fe[(f, e)] = fe_count[(f, e)] / f_count[f]

	iter_num += 1
	if iter_num > 15:
		break
################################  basckward e -> f  ##################################

# initialization
p_ef = defaultdict(int)
for (f_i, e_j) in fe_count_init.keys():
	p_ef[(f_i,e_j)] = 1. 

iter_num = 0

while True:
	e_count = e_count_init.copy()
	ef_count = fe_count_init.copy()
	for (k , (f, e)) in enumerate(bitext):
		for f_i in f:
			z = 0
			for e_j in e:
				z += p_ef[(f_i,e_j)]
			for e_j in e:
				c = p_ef[(f_i,e_j)] / z
				ef_count[(f_i,e_j)] += c
				e_count[e_j] += c
		if k % 5000 == 0:
			sys.stderr.write(".")
	sys.stderr.write("\n")

	for (f, e) in ef_count.keys():
		p_ef[(f, e)] = ef_count[(f, e)] / e_count[e]

	iter_num += 1
	if iter_num > 15:
		break

##############################        Union         ############################
# p = defaultdict(int)
# for fe in p_fe:
# 	if fe in p_ef:
# 		p[fe] = p_fe[fe]

##############################  output the alignment  ############################

for (f, e) in bitext:
	for (j, e_j) in enumerate(e): 
		best_p_fe = 0
		best_i_fe = 0
		best_p_ef = 0 
		best_i_ef = 0
		for (i, f_i) in enumerate(f):
			if p_fe[(f_i,e_j)] >= best_p_fe:
				best_i_fe = i
				best_p_fe = p_fe[(f_i,e_j)]
			if p_ef[(f_i,e_j)] >= best_p_ef:
				best_i_ef = i
				best_p_ef = p_ef[(f_i,e_j)]
		for (i, f_i) in enumerate(f):
			if (p_fe[(f_i,e_j)] == best_p_fe or p_ef[(f_i,e_j)] == best_p_ef)\
			and (p_fe[(f_i,e_j)] >= opts.threshold or p_ef[(f_i,e_j)] >= opts.threshold):
				sys.stdout.write("%i-%i " % (i,j))
	sys.stdout.write("\n")


# for (f, e) in bitext:
# 	for (i, f_i) in enumerate(f): 
# 		for (j, e_j) in enumerate(e):
# 			print p_fe[(f_i,e_j)],p_ef[(f_i,e_j)]
# 			if p_fe[(f_i,e_j)] >= opts.threshold or p_ef[(f_i,e_j)] >= opts.threshold:
# 				sys.stdout.write("%i-%i " % (i,j))
# 	sys.stdout.write("\n")