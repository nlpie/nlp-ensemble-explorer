#!/usr/bin/env python
# coding: utf8
"""
usage:  ipython spacy_phrase_matcher.py [-h] [-n 10000] [-l en] patterns_loc text_loc
e.g.,  
ipython /mnt/DataResearch/gsilver1/scripts/spacy_phrase_matcher.py /mnt/DataResearch/DataStageData/ed_provider_notes/methods_paper/analysis/misc/master_symptoms_gazetteer.txt /mnt/DataResearch/DataStageData/ed_provider_notes/methods_paper/notes_for_analysis/

Match a large set of multi-word expressions in O(1) time.

The idea is to associate each word in the vocabulary with a tag, noting whether
they begin, end, or are inside at least one pattern. An additional tag is used
for single-word patterns. Complete patterns are also stored in a hash set.
When we process a document, we look up the words in the vocabulary, to
associate the words with the tags.  We then search for tag-sequences that
correspond to valid candidates. Finally, we look up the candidates in the hash
set.

For instance, to search for the phrases "Barack Hussein Obama" and "Hilary
Clinton", we would associate "Barack" and "Hilary" with the B tag, Hussein with
the I tag, and Obama and Clinton with the L tag.

The document "Barack Clinton and Hilary Clinton" would have the tag sequence
[{B}, {L}, {}, {B}, {L}], so we'd get two matches. However, only the second
candidate is in the phrase dictionary, so only one is returned as a match.

The algorithm is O(n) at run-time for document of length n because we're only
ever matching over the tag patterns. So no matter how many phrases we're
looking for, our pattern set stays very small (exact size depends on the
maximum length we're looking for, as the query language currently has no
quantifiers).

The example expects a .bz2 file from the Reddit corpus, and a patterns file,
formatted in jsonl as a sequence of entries like this:

NB: change bz2 input to glob directory -> gms-sept-30-2020

{"text":"Anchorage"}
{"text":"Angola"}
{"text":"Ann Arbor"}
{"text":"Annapolis"}
{"text":"Appalachia"}
{"text":"Argentina"}

Reddit comments corpus:
* https://files.pushshift.io/reddit/
* https://archive.org/details/2015_reddit_comments_corpus

Compatible with: spaCy v2.0.0+
"""
from __future__ import print_function, unicode_literals, division

#from bz2 import BZ2File
import time
import plac
import json
import glob
import os

from spacy.matcher import PhraseMatcher
import spacy


@plac.annotations(
    patterns_loc=("Path to gazetteer", "positional", None, str),
    text_loc=("Path to corpus files", "positional", None, str),
    n=("Number of texts to read", "option", "n", int),
    lang=("Language class to initialise", "option", "l", str),
)
def main(patterns_loc, text_loc, n=10000, lang="en"):
    nlp = spacy.blank(lang)
    nlp.vocab.lex_attr_getters = {}
    phrases = read_gazetteer(nlp.tokenizer, patterns_loc)
    count = 0
    t1 = time.time()
    for ent_id, text, fname in get_matches(nlp.tokenizer, phrases, read_text(text_loc, n=n)):
        print(json.dumps({"symptom": text, "note": fname}))
        count += 1
    t2 = time.time()
    print("%d docs in %.3f s. %d matches" % (n, (t2 - t1), count))


def read_gazetteer(tokenizer, loc, n=-1):
    for i, line in enumerate(open(loc)):
        data = json.loads(line.strip())
        phrase = tokenizer(data["text"].lower())
        for w in phrase:
            _ = tokenizer.vocab[w.text]
        if len(phrase) >= 1:
            yield phrase


# change to glob
def read_text(data_folder, n=10000):
    for fname in glob.glob(data_folder + "*.txt"):
        file = os.path.basename(fname)
        u = file.split('.')[0]
        with open(fname) as fn:
            Lines = fn.readlines()
            i = 0
            for line in Lines:
            #for i, line in enumerate(Lines):
                i+=0
                yield line.lower(), file
                #if i >= n:
                #    break
            

def get_matches(tokenizer, phrases, texts):
    matcher = PhraseMatcher(tokenizer.vocab)
    matcher.add("Phrase", None, *phrases)
    for text, fname in texts:
        doc = tokenizer(text)
        for w in doc:
            _ = doc.vocab[w.text]
        matches = matcher(doc)
        for ent_id, start, end in matches:
            yield (ent_id, doc[start:end].text, fname)


if __name__ == "__main__":
    if False:
        import cProfile
        import pstats

        cProfile.runctx("plac.call(main)", globals(), locals(), "Profile.prof")
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats()
    else:
        plac.call(main)
