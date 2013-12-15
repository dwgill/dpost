'''
Created on Aug 22, 2013

@author: tvandrun, dwgill
'''


from __future__ import division
import nltk
from nltk.corpus import PlaintextCorpusReader
import nltk.tokenize
from nltk import FreqDist
from collections import defaultdict
import sys
import operator
from itertools import izip

res = './res/'
training_corpus = 'wonderland.txt'
testing_corpus  = 'looking-glass.txt'
# training_corpus = '*.txt'


# 1. Load a (training) corpus.
# In the code below, the corpus will be
# referred to by variable all_text
all_text = PlaintextCorpusReader(res, r'\w+\.txt').words()

# make the training text lowercase
all_text_lower = (x.lower() for x in all_text)
freq_dist = FreqDist(all_text_lower)

# make a reduced vocabulary (here, 500 types)
vocab = freq_dist.keys()[:500]
vocab = vocab + ["***"]

# 2. Make a reduced form of the PennTB tagset

tg_formating = {'N':'noun-like','V':'verb-like','AJ':'adjective-like',
        'AV':'adverb-like','I':'interjection','S':'symbol','G':'grouping',
        'E':'end-of-sentence', 'UN':'unknown'}

penntb_to_reduced = {}
# noun-like
for x in ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'EX', 'WP', 'FW', 'UH'] :
    penntb_to_reduced[x] = 'N'
# verb-like
for x in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'MD', 'TO'] :
    penntb_to_reduced[x] = 'V'
# adjective-like
for x in ['POS', 'PRP$', 'WP$', 'JJ', 'JJR', 'JJS', 'DT',
        'CD', 'PDT', 'WDT', 'LS']:
    penntb_to_reduced[x] = 'AJ'
# adverb-like
for x in ['RB', 'RBR', 'RBS', 'WRB', 'RP', 'IN', 'CC']:
    penntb_to_reduced[x] = 'AV'
# groupings
for x in ['\'\'', '(', ')', ',', ':', '``', 'SYM', '$', '#'] :
    penntb_to_reduced[x] = 'G'
# end-of-sentence symbols
penntb_to_reduced['.'] = 'E'
# unidentified phenomena

reduced_tgs = ['N', 'V', 'AJ', 'AV', 'G', 'E']

# 3. tag the corpus
all_tagged = nltk.pos_tag(all_text)

# 4. make the probability matrices

# a tally from types to tags; a tally from tags to next tags
tg_word_tally = {tg:{word:1 for word in vocab} for tg in reduced_tgs}

tg_trans_tally = {prev_tg:{current_tg:1 for current_tg
    in reduced_tgs} for prev_tg in reduced_tgs}

tg_totals = {tg:1 for tg in reduced_tgs}

previous_tg = 'E' # "ending" will be the dummy initial tag
for word, current_tg in all_tagged:
    word = word.lower()

    if current_tg == '-NONE-':
        if word[0] == '.' or word[0] == '!' or word[0] == '?':
            current_tg = 'E'
        else:
            current_tg = 'G'
    else:
        current_tg = penntb_to_reduced[current_tg]

    if word in vocab:
        tg_word_tally[current_tg][word] += 1

    tg_totals[current_tg] += 1
    tg_trans_tally[previous_tg][current_tg] += 1

    previous_tg = current_tg


# LaPlace Smoothing
K = 1

# now, make the actual transition probability matrices 

trans_probs = {i_tg: defaultdict(lambda: K / (tg_totals[i_tg] + K *
    len(tg_word_tally[i_tg].values()))) for i_tg in reduced_tgs}

for i_tg in reduced_tgs:
    for f_tg in reduced_tgs:
        trans_probs[i_tg][f_tg] = tg_trans_tally[i_tg][f_tg] / tg_totals[i_tg]



emit_probs = {i_tg: defaultdict(lambda: K / (tg_totals[i_tg] +
    K * len(tg_word_tally[i_tg].values()))) for i_tg in reduced_tgs}

for tg in reduced_tgs:
    for wd in vocab:
        emit_probs[tg][wd] = tg_word_tally[tg][wd] / tg_totals[tg]


def pos_tagging(sequence, trans_probs, emit_probs, vocab, tags, start):
    viterbi = [{}]
    best_tg_sqs = {}

    for tg in tags:
        viterbi[0][tg] = trans_probs[start][tg] * emit_probs[tg][sequence[0]]
        best_tg_sqs[tg] = [tg]

    def eval_prev(curr_tg, prev_tg, index):
        indie_prob = viterbi[index - 1][prev_tg]
        trans_prob = trans_probs[prev_tg][curr_tg]
        emit_prob  = emit_probs[curr_tg][sequence[index]]
        return (indie_prob * trans_prob * emit_prob), prev_tg

    for index in range(1, len(sequence)):
        viterbi.append({})
        new_best_tg_sqs = {}
        for tg in tags:
            candidates = (eval_prev(tg, prev_tg, index)
                    for prev_tg in tags)
            prob_given_prev, prev_tg = max(candidates,
                    key=operator.itemgetter(0))
            viterbi[index][tg] = prob_given_prev
            new_best_tg_sqs[tg] = best_tg_sqs[prev_tg] + [tg]
        best_tg_sqs = new_best_tg_sqs

    final_candidates = ((viterbi[index][tg], tg) for tg in tags)
    _, best_tg = max(final_candidates, key=operator.itemgetter(0))

    return best_tg_sqs[best_tg]

# 6. try it out: run the algorithm on the test data
tokenizer = nltk.tokenize.TreebankWordTokenizer()

def tag_string_raw(string):
    sequence = tokenizer.tokenize(string.strip().lower())
    return pos_tagging(sequence, trans_probs, emit_probs, vocab,
            reduced_tgs, 'E')

def tag_string_nltk_raw(string):
    sequence = tokenizer.tokenize(string.strip().lower())
    nltk_result = (penntb_to_reduced[x[1]] for x in nltk.pos_tag(sequence))
    return nltk_result

def tag_string(string):
    sequence = tokenizer.tokenize(string.strip().lower())
    my_result = pos_tagging(sequence, trans_probs, emit_probs, vocab,
            reduced_tgs, 'E')

    nltk_result = (penntb_to_reduced[x[1]] for x in nltk.pos_tag(sequence))
    line_frmt = '{word}:\t{tag} ({tag_big})'
    my_result_l = ['dwgill results:']
    nltk_result_l = ['nltk results:']

    my_result_l.extend(line_frmt.format(word=word,tag=tag,
        tag_big=tg_formating[tag])
            for (word, tag) in izip(sequence,my_result))
    nltk_result_l.extend(line_frmt.format(word=word,tag=tag,
        tag_big=tg_formating[tag])
            for (word, tag) in izip(sequence,nltk_result))

    print '\n\t'.join(my_result_l)
    print '\n'
    print '\n\t'.join(nltk_result_l)
