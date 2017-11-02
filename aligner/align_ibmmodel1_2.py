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
q = defaultdict(int)
q_ef = defaultdict(int)

for (n, (f, e)) in enumerate(bitext):
	for (i, f_i) in enumerate(f):
		f_count_init[f_i] = 0
		for (j, e_j) in enumerate(e):
			fe_count_init[(f_i,e_j)] = 0
			q[(i, j, len(f), len(e))] = 1.
			q_ef[(i, j, len(e), len(f))] = 1.
	for (j, e_j) in enumerate(e):
		e_count_init[e_j] = 0	
	if n % 500 == 0:
		sys.stderr.write(".")

iter_max =15
################################  forward f -> e  ##################################

# initialization
p = defaultdict(int)
for (f_i, e_j) in fe_count_init.keys():
	p[(f_i,e_j)] = 1. 

iter_num = 0

while iter_num < iter_max :

	f_count = f_count_init.copy()
	fe_count = fe_count_init.copy()
	ijlm_count = defaultdict(int)
	jlm_count = defaultdict(int)

	for (k , (f, e)) in enumerate(bitext):
		for (j,e_j) in enumerate(e):
			z = 0
			for (i,f_i) in enumerate(f):
				z += q[(i, j, len(f), len(e))] * p[(f_i, e_j)]
			for (i,f_i) in enumerate(f):

				delta = q[(i, j, len(f), len(e))] * p[(f_i, e_j)] / z

				fe_count[(f_i,e_j)] += delta
				f_count[f_i] 		+= delta
				ijlm_count[(i, j, len(f), len(e))]  += delta
				jlm_count[(j, len(f), len(e))]		+= delta	

		if k % 5000 == 0:
			sys.stderr.write(".")
	sys.stderr.write("\n")
	
	del delta

	for (f, e) in fe_count.keys():
		p[(f, e)] = fe_count[(f, e)] / f_count[f]
	for (i,j,l,m) in ijlm_count.keys():
		q[(i,j,l,m)] = ijlm_count[(i,j,l,m)] / jlm_count[(j,l,m)]

	iter_num += 1

################################  basckward e -> f  ##################################

# initialization
p_fe = defaultdict(int)
for (f_i, e_j) in fe_count_init.keys():
	p_fe[(f_i,e_j)] = 1. 

iter_num = 0

while iter_num < iter_max :
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


##############################  output the alignment  ############################

for (f, e) in bitext:
	for (j, e_j) in enumerate(e): 
		best_p_fe = 0
		best_i_fe = 0
		best_p_ef = 0 
		best_i_ef = 0
		for (i, f_i) in enumerate(f):
			if p[(f_i,e_j)] * q[i,j,len(f),len(e)] >= best_p_fe:
				best_i_fe = i
				best_p_fe = p[(f_i,e_j)]
			if p_fe[(f_i,e_j)]  >= best_p_ef:
				best_i_ef = i
				best_p_ef = p_fe[(f_i,e_j)]
		for (i, f_i) in enumerate(f):
			if p[(f_i,e_j)] * q[i,j,len(f),len(e)] == best_p_fe or \
				p_fe[(f_i,e_j)] == best_p_fe  \
			and p[(f_i,e_j)] * q[i,j,len(f),len(e)] >= opts.threshold:# \
			# 	p_ef[(f_i,e_j)] * q_ef[i,j,len(e),len(f)] >= opts.threshold):
				sys.stdout.write("%i-%i " % (i,j))
	sys.stdout.write("\n")

# for (f, e) in bitext:
# 	for (i, f_i) in enumerate(f): 
# 		for (j, e_j) in enumerate(e):
# 			if p[(f_i,e_j)] * q[i,j,len(f),len(e)] >= opts.threshold:
# 				sys.stdout.write("%i-%i " % (i,j))
# 	sys.stdout.write("\n")