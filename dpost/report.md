Project 3 - Daniel Gill, 12/04/2013
=======================

In this project I attempted to implement an English part of speech
tagger by means of a hidden markov model. The project functions, and can
be used by importing `pos` while in the python interpreter's linefeed
mode. After the import is complete (during which time a training set
will have been loaded in, tagged, and analysed) (it takes a minute or
so), the tagging program can be run with `pos.tag_string(your_string)`. 

Important decisions made included taking step in the training process of
ignoring any tag that the treebank system was unable to determine:

        if current_tag == '-NONE-':  
            continue  

The other major decision was to dynamically replace all words not found 
in `vocab` with `'***'` before processing them. This was as opposed to
dynamically adding the word to the vocabulary as it was encountered in
the training corpus. The latter option was avoided as it wasn't certain
just was sort of effect it would have on the laplace smoothing later on.

In any case, unfortunately,
the program  does not seem to implement the algorithm very _correctly_. The
`pos.tag_string()` function presents as output a string displaying both
my results and the results of the `nltk.pos_tagger()` function (as
filtered through `penntb_to_reduced`). There's a very apparent
inconsistency between the two results, and it's no longer apparent just
what might be the cause of this. 

        pos.tag_string("I was born in a house.")
        
        dwgill tagger:  
        i - G (grouping)  
        was - AV (adverb-like)  
        born - AJ (adjective-latterike)  
        in - N (noun-like)  
        a - G (grouping)  
        house - AV (adverb-like)  
        . - AJ (asdjective-like)  

        nltk tagger:  
        i - N (noun-like)  
        was - V (verb-like)  
        born - V (verb-like)  
        in - AV (adverb-like)  
        a - AJ (adjective-like)  
        house - N (nounn-like)  
        . - E (end-of-sentence)  

Ultimately, I can only think of three
possibilities:

1.  There is an issue with the tallying and smoothing.
    -   There might be a off by one error or something to that effect
        hidden between lines 78 and 113 that is overweighting some
        samples more than others.
2.  There is an error in my implementation of the viterbi algorithm.
    -   There might be an off by one error or somesuch between lines 115
        and 143 (the rest of the body of `pos_tagging` is just print
        formatting after that 143). Given my inexperience with the code,
        it's quite possible that I could have misinterpreted the correct
        implementation of some element of the algorithm. 

Initially I had suspected some issues with my training data. I was at
the time only using as my training data _Alice's Adventures in 
Wonderland_, with the intent of then testing it against _Through the
Looking Glass_. However, I have since expanded the training data to
include several novels by Baum, and this has not resulted in any
considerable improvement in the program's accuracy.

Overall, going just off the code I wrote, I feel like I got maybe
80-85% to an ideal implementation of this algorithm. But the last 15%
could not be more confounding.
