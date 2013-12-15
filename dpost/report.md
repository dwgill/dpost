Project 4 - Daniel Gill, 12/13/2013
===================================
The aim of this project was to improve the correctness of my previous
implementation of an HMM-based part-of-speech tagger. I identified three
possible areas of failure in the original:

1.  Issues with my selection of possible parts of speech. i.e. my set of
    possible states was poorly calibrated.
2.  Logic errors in the smoothing process.
3.  Logic errors in my implementation of the Viterbi algorithm.

Lacking the time to carefully test any of these areas individually, I 
opted to address all of them simultaniously.

####Possible Parts of Speech####
Under the recommendation of Tim, I removed the 'symbol' and 
'interjection' states and merged them into 'noun-like-things' 
and 'groupings' states respectively. In most corpuses these srates were being 
recorded exceedingly rarely, which apparently was causing some problems with 
the way Laplacing Smoothing was implemented.

####Smoothing Process####
Initially I tried to 'manually' smooth the data after it was collected.
This ended up being hard to read/comprehend and therefore maintain, so
the algorithm was rewritten from scratch. Beforehand, the dictionaries
containing the transmission probabilities were initialized to
0 and the smoothing was left for later, whereas now they are initialized
to `K / (tg_totals[i_tg] + K * len(tg_word_tally[i_tg].values()))`
where `i_tg` is the initial tag in a state transition, and
`tg_word_tally` is a dict of dicts such that `tg_word_tally[A][B]` is
the number of times `A` has been observed to transition to `B`. This is
then overwritten for any transitions we have observed (see lines
117-119), but it does mean that any sort of transition conceivable does
now have a computed (smoothed) probability.

(Emission probabilities is a similar story.)

####Viterbi Algorithm####
The viterbi algorithm implementation has been entirely scrapped and
rewritten. This, too, was ultimately an exercise of making more concise
and (hopefully) readable code. The biggest change is that backpointer
went from being a list of dictionaries to a dictionary of lists, and I
was able to elemenate the need to laboriously backtrack through it to
reconstruct the the ideal sequence, and instead keep an ideal sequence
for each given previous state.

Thanks is again due to Tim for suggesting how to handle the max/argmax
computations. The `operator` module is truly a realm of wizardry.

####Outcome####
All in all, the tagger seems to preform much better than my previous
implementation, ultimately almost reaching parity with the default nltk
tagger.

**Recommended sentences to tag:**
>   _"These are the times that try mens' souls."_

Observe how my own implementation still differs from the nltk solution 
in dealing with apostrophes.

>  _"Call me Ishmael"_

Observe that both the nltk tagger and my own solution struggle to 
figure out what part of speech 'Ishmael' is.

>   _"Alice was beginning to get very tired of sitting by
>   her sister on thebank, and of having nothing to do: once or twice
>   she had peeped into thebook her sister was reading, but it had no
>   pictures or conversations init, 'and what is the use of a book,' thought
>   Alice 'without pictures orconversation?'"_

This one is a bit more involved. A full comparison is listed below.

    dwgill: N, V, V, V, V, AV, N, AV, V, AV, AJ, N, AV,  
    nltk:   N, V, V, V, V, AV, V, AV, N, AV, AJ, N, AV,

    dwgill: AJ, N, G, AV, AV, V, N, V, V, G, AV, AV, AJ,  
    nltk:   AJ, N, G, AV, AV, V, N, V, V, G, AV, AV, N,

    dwgill: N, V, V, AV, AJ, N, AJ, N,  V, N, G, AV, N,  
    nltk:   N, V, V, AV, AJ, N, N,  AJ, V, V, G, AV, N,

    dwgill: V, AJ, N, AV, N, AV, N, G, AV, N, V, AJ, N,  
    nltk:   V, AJ, N, AV, N, AV, N, G, N,  N, V, AJ, N,

    dwgill: AV, AJ, N, G, G, V, AV, AJ, N, AV, N, E, G  
    nltk:   AV, AJ, N, G, G, V, N,  AV, N, AV, N, E, G

All in all, a drastic improvement over previous standards.

####Usage####
Simply import `pos` into python in an interpreter session. After it's
finished loading, you can detailed comparisons with
`pos.tag_string(your_string)`, and raw state fields with
`pos.tag_string_raw(your_string)` and
`pos.tag_string_nltk_raw(your_string)`
