
#from nltk.corpus import words
import tensorflow as tf 
import numpy as np
from utilities.tokenizer import MyTokenizer
import re

from collections import Counter
import itertools

regexps = {
    '<PHN>' : r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4})',
    '<SSN>' : r'\d{3}-?\d{2}-?\d{4}',
    '<DATE>' : r'\d{1,2}[-\.\s\/]\d{1,2}[\.\s\/]\d{1,4}',
    '<EMAIL>' : r'[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}',
    '<TIME>' : r'([0-9]|0[0-9]|1[0-9]|2[0-3])(:[0-5][0-9])(:[0-5][0-9])?(\s*[AaPp][Mm])'
}

symbols = {'*', '=', '~', '^', '\x7f', '\x1a', '�', '\\', '|'}

def apply_regex(string, regexps):
    new_string = string

    for pii, r in regexps.items():
        targets = re.findall(r, new_string)
        for t in targets:
            try:
                new_string = new_string.replace(t, ' ' + pii + ' ')
            except:
                new_string = new_string.replace(''.join(t), ' ' + pii + ' ')

    return new_string

def prepare_text_training(free_text, mask_pii = False, limit = 200):

    tk = MyTokenizer()
    corpus = []
    tokenized_corpus = []
    progbar = tf.keras.utils.Progbar(len(free_text))
    for x in free_text:
        try:
            y = x
            for symbol in symbols:
                y = y.replace(symbol, '')
            z = y
            y = y.lower()
            if mask_pii:
                y = apply_regex(y, regexps)
            y = tk.tokenize(y)
            if len(y) <= limit:
                tokenized_corpus.append(y)
                corpus.append(z)
        except:
            pass
        progbar.add(1)
    return tokenized_corpus, corpus

def flatten(l):
    return list(itertools.chain.from_iterable(l))

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq 

def create_masking_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 4), tf.float32)
    return seq
   
def create_masks(inp):
    enc_padding_mask = create_padding_mask(inp)
    language_mask = create_masking_mask(inp)
    return enc_padding_mask, language_mask