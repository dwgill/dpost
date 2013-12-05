'''
Created on Aug 22, 2013

@author: tvandrun, dwgill
'''


import nltk
from nltk.corpus import PlaintextCorpusReader
import nltk.tokenize
from nltk import FreqDist
from collections import defaultdict
import sys

res = './res/'
training_corpus = 'wonderland.txt'
testing_corpus  = 'looking-glass.txt'
# training_corpus = '*.txt'

# LaPlace Smoothing
K = 1

# 1. Load a (training) corpus.
# In the code below, the corpus will be
# referred to by variable all_text
all_text = PlaintextCorpusReader(res, r'\w+\.txt').words()

# make the training text lowercase
all_text_lower = (x.lower() for x in all_text)
freq_dist = FreqDist(all_text_lower)

# make a reduced vocabulary (here, 500 types)
vocab = freq_dist.keys() # [:500]
vocab = vocab + ["***"]

# 2. Make a reduced form of the PennTB tagset

tag_formating = {'N':'noun-like','V':'verb-like','AJ':'adjective-like',
        'AV':'adverb-like','I':'interjection','S':'symbol','G':'grouping',
        'E':'end-of-sentence', 'UN':'unknown'}

penntb_to_reduced = {}
# noun-like
for x in ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'EX', 'WP'] :
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
# interjections
for x in ['FW', 'UH'] :
    penntb_to_reduced[x] = 'I'
# symbols
for x in ['SYM', '$', '#'] :
    penntb_to_reduced[x] = 'S'
# groupings
for x in ['\'\'', '(', ')', ',', ':', '``'] :
    penntb_to_reduced[x] = 'G'
# end-of-sentence symbols
penntb_to_reduced['.'] = 'E'
# unidentified phenomena

reduced_tags = ['N', 'V', 'AJ', 'AV', 'I', 'S', 'G', 'E']

# 3. tag the corpus
all_tagged = nltk.pos_tag(all_text)

# 4. make the probability matrices

# a tally from types to tags; a tally from tags to next tags
# For now, these are objective.
tag_word_tally = {tag:{word:0 for word in vocab} for tag in reduced_tags}
tag_transition_tally = {prev_tag:{current_tag:0
    for current_tag in reduced_tags} for prev_tag in reduced_tags}

previous_tag = 'E' # "ending" will be the dummy initial tag
for word, current_tag in all_tagged:
    if current_tag == '-NONE-':
        continue

    word = word.lower()
    current_tag = penntb_to_reduced[current_tag]
    if word not in tag_word_tally:
        word = '***'
    tag_word_tally[current_tag][word] += 1
    tag_transition_tally[previous_tag][current_tag] += 1
    previous_tag = current_tag
tag_transition_tally[previous_tag]['E'] += 1

# now, make the actual transition probability matrices 
trans_probs = {}
for i_tag in reduced_tags:
    i_tag_count = sum(tag_transition_tally[i_tag].values())
    # LaPlace smoothing---add the fabricated samples to the total.
    i_tag_count_s = i_tag_count + len(reduced_tags) * K
    for t_tag in reduced_tags:
        it_count = tag_transition_tally[i_tag][t_tag]
        # LaPlace smoothing---add a fabricated sample
        it_count_s = it_count + K
        trans_probs[(i_tag, t_tag)] = it_count_s / float(i_tag_count_s)

emit_probs = {}
for tag, counts_dict in tag_word_tally.iteritems():
    t_count_s = sum(counts_dict.values()) + K * len(counts_dict)
    for word, tw_count in counts_dict.iteritems():
        tw_count_s = tw_count + K
        emit_probs[(tag,word)] = tw_count_s / float(t_count_s)

def pos_tagging(sequence, trans_probs, emit_probs, vocab, tags, start) :
    viterbi = {tag:[0.0 for item in sequence] for tag in tags}
    backpointer = {tag: [start for item in sequence] for tag in tags}


    # Initialization
    for tag in tags:
        viterbi[tag][0] = trans_probs[(start, tag)] * emit_probs[(tag, sequence[0])]
        backpointer[tag][0] = start

    def determine_backpointer(tag, index):
        key = lambda prev_tag: viterbi[prev_tag][index - 1] * trans_probs[(prev_tag, tag)]
        max_prev_tag = max(tags, key=key)
        backpointer[tag][index] = max_prev_tag

    def determine_prob_given_hist(tag, index):
        word = sequence[index] if sequence[index] in vocab else '***'
        key = lambda prev_tag: (viterbi[prev_tag][index - 1] * trans_probs[(prev_tag, tag)]
                * emit_probs[(tag, word)])
        max_prob = max(key(prev_tag) for prev_tag in tags)
        viterbi[tag][index] = max_prob

    for index in range(1, len(sequence)):
        for tag in tags:
            determine_backpointer(tag, index)
            determine_prob_given_hist(tag, index)

    last_index = len(sequence) - 1
    last_tag = max(tags, key = lambda tag: viterbi[tag][last_index])

    tagged = []
    for index in range(last_index, -1, -1):
        next_word = "{word}\t\t{tag} ({qtag})".format(word=sequence[index],
                tag=last_tag, qtag=tag_formating[last_tag])
        tagged.append(next_word)
        last_tag = backpointer[last_tag][index]
    tagged.append('dwgill tagger:')
    tagged.reverse()
    tagged.append('\nnltk tagger:')

    for word,tag in nltk.pos_tag(sequence):
        tag = penntb_to_reduced[tag] if tag in penntb_to_reduced else '??'
        tagged.append("{word}\t\t{tag} ({qtag})".format(word=word,tag=tag,
            qtag=tag_formating[tag]))
    return '\n'.join(tagged)

# 6. try it out: run the algorithm on the test data
tokenizer = nltk.tokenize.TreebankWordTokenizer()

def tag_string(string):
    sequence = tokenizer.tokenize(string.strip().lower())
    print pos_tagging(sequence, trans_probs, emit_probs, vocab,
            reduced_tags, 'E')
