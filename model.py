import os
import json

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa
import numpy as np
import random

from collections import Counter
import itertools

from encoder import ELMoEncoder
from optimizer.custom_schedule import CustomSchedule
from utilities.utils import create_padding_mask

MAX_VOCAB_SIZE = 125000
MAX_N_CHARS = 500

class MyELMo:

    def __init__(self, parameters, path_model):
        
        """
        Parameters
        ----------
        
        parameters: dict
            Dictionary of parameters to initialize the model.
            
        """
        
        self.path_model = path_model
        self.n_special_tokens = 5 # Padding / [CLS] / [SEP] / [UNKNOWN] / [MASKED]
        self.n_special_characters = 7 # Padding /  [BOW] / [EOW] / [CLS] / [SEP] / [MASKED] / [PADDING]
        
        self.d_model = parameters['d_model']
        self.num_layers = parameters['num_layers']
        self.pe_input = parameters['pe_input']
        self.filters = parameters['filters']
        self.d_embeddings = parameters['d_embeddings']
        self.num_highway_layers = parameters['num_highway_layers']
  
        self.fitted = False

        #try:
         
        if parameters['fitted'] == True:
            self.fitted = True
            self.word2idx = parameters['word2idx']
            self.char2idx = parameters['char2idx']
            self.idx2word = {v: k for k, v in self.word2idx.items()}
            self.vocabulary = set(parameters['vocabulary'])
            self.list_vocabulary = parameters['vocabulary']
            self.vocab_size = parameters['vocab_size']


        self.generator = ELMoEncoder(vocab_size = MAX_VOCAB_SIZE,
                                    n_chars = MAX_N_CHARS,
                                    filters = self.filters,
                                    output_dim = MAX_VOCAB_SIZE,
                                    d_embeddings = self.d_embeddings,
                                    hidden_size = self.d_model,
                                    num_layers = self.num_layers,
                                    num_highway_layers = self.num_highway_layers,
                                    max_sequence_length = self.pe_input,
                                    output_dropout = 0.5,
                                    recurrent_dropout = 0.5)

        # Optimizer

        gen_learning_rate = CustomSchedule()
        self.generator_optimizer = tfa.optimizers.AdamW(
                                                weight_decay = 5e-4,
                                                learning_rate = gen_learning_rate,
                                                beta_1 = 0.9,
                                                beta_2 = 0.999,
                                                epsilon = 1e-06)

        self.gen_decay_var_list	= [v for v in self.generator.variables[:-2] if 'layer_norm' not in v.name and 'bias' not in v.name]

        self.ckpt = tf.train.Checkpoint(generator = self.generator, 
                                        generator_optimizer = self.generator_optimizer)
        checkpoint_path = os.path.join(self.path_model, 'tf_ckpts')
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, checkpoint_path, max_to_keep=1)

        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!')

    def get_index_word(self, word):

        """
        
        Returns the index of a token. Returns 3 (unknown) if it is an out-of-vocabulary token.
        """
        
        if word in self.vocabulary:
            return self.word2idx[word]
        else:
            if word == '[CLS]':
                return 1
            elif word == '[SEP]':
                return 2
            elif word == '[MASKED]':
                return 4
            else:
                return 3 # index for UNKNOWN
    
    def get_word_index(self, index):

        if index >= self.n_special_tokens and index <= self.vocab_size - 1 + self.n_special_tokens:
            return self.idx2word[index]
        elif index == 1:
            return '[CLS]'
        elif index == 2:
            return '[SEP]'
        elif index == '[MASKED]':
            return 4
        else:
            return '[UNKNOWN]'

        
    def pad_char(self, sentence, max_len, max_len_char):
        
        """
        Parameters
        ----------
        sentence: list of int
        max_len: int
            Maximum sentence length in a batch.
        max_len_char: int
            Maximum word length in a batch
        The output has a shape (max_len, max_len_char).
        Output
        ------
        Returns the padded characters' indexes.
        """

        def get_index_char(x):
            def index2char(char):
                try:
                    return self.char2idx[char]
                except:
                    return 6
            if x == '[CLS]':
                return [1, 3, 2]
            elif x == '[SEP]':
                return [1, 4, 2]
            elif x == '[MASKED]':
                return [1, 5, 2]
            else:
                return [1] + [index2char(c) for c in x] + [2]
        indexed_char = list(map(get_index_char, sentence))
        chars = tf.keras.preprocessing.sequence.pad_sequences(indexed_char, maxlen = max_len_char, padding = 'post', truncating = 'post')
        padded_chars = tf.concat([chars, tf.constant(0., shape = (max_len - chars.shape[0], max_len_char))], axis = 0)
        return padded_chars

    def process_sentence(self, tar_sentence, tar_indexes):
        
        sentence = ['[CLS]']
        indexes = [1]
        sentence += tar_sentence
        indexes += tar_indexes
        sentence.append('[SEP]')
        indexes.append(2)

        return sentence, indexes
        
    def get_next_batch(self, batch_size, set_index, source_text, indexed_text):

        num_samples = np.minimum(batch_size, len(set_index))
        target_indexes = random.sample(set_index, num_samples)
        set_index.difference_update(set(target_indexes))

        tar_text = source_text[target_indexes]
        tar_indexes = indexed_text[target_indexes]
        temp = list(map(lambda x: self.process_sentence(tar_text[x], tar_indexes[x]), range(num_samples)))
        inp_text, inp_indexes = list(zip(*temp))

        max_len_char = 25 
        max_len = self.pe_input

        inp_chars = list(map(lambda x: self.pad_char(x, max_len, max_len_char)[tf.newaxis, :, :], inp_text))
        inp_chars = tf.concat(inp_chars, axis = 0)

        inp_words = tf.cast(tf.keras.preprocessing.sequence.pad_sequences(inp_indexes, maxlen = max_len, padding = 'post'), tf.int32)

        return inp_words, inp_chars

    @tf.function
    def train_step_generator(self, inp_words, inp_chars):

        enc_padding_mask = create_padding_mask(inp_words)
        mask = 1. - enc_padding_mask

        with tf.GradientTape() as tape:
            outputs_encoder = self.generator([inp_chars, enc_padding_mask], training = True)
            forward_logits = outputs_encoder['forward_logits']
            backward_logits = outputs_encoder['backward_logits']

            mask_logits = tf.concat([tf.zeros(self.n_special_tokens + self.vocab_size), tf.ones(MAX_VOCAB_SIZE  - self.n_special_tokens - self.vocab_size)], 0)
            forward_logits += mask_logits[tf.newaxis, tf.newaxis, :] * (-1e9)
            backward_logits += mask_logits[tf.newaxis, tf.newaxis, :] * (-1e9)

            forward_loss = tf.keras.losses.sparse_categorical_crossentropy(inp_words[:, 1:], forward_logits[:, :-1, :], from_logits = True)
            backward_loss = tf.keras.losses.sparse_categorical_crossentropy(inp_words[:, :-1], forward_logits[:, 1:, :], from_logits = True)
            loss = forward_loss + backward_loss

            loss = tf.math.divide_no_nan(tf.reduce_sum(loss * mask[:, :-1], axis = 1), tf.reduce_sum(mask - 1., axis = 1))
            batch_loss = tf.reduce_mean(loss)

            variables = self.generator.trainable_variables[:-2]
            gradients = tape.gradient(batch_loss, variables)    
            self.generator_optimizer.apply_gradients(zip(gradients, variables), decay_var_list = self.gen_decay_var_list)
        
        return batch_loss
               
    def fit(self, corpus, epochs, batch_size, min_size = 10, min_count = 1, STORAGE_BUCKET = None):
        
        """
        Fits the model:
            - Initializes or updates the mapping words / indexes and the mapping characters / indexes
            - Processes the text data and arranges it to create batch of sentence with similar lengths
            - Performs the stochastic gradient descent
            
        corpus: list
            List of tokenized sentences.
            
        epochs: int
            Number of epochs.
        
        batch_size: int
            Batch size.
            
        window_size: int
            Window size in the Word2Vec model used to initialize the embedding matrices of the different embedding layers.
            If None, the embedding layers are randomly initialized.
        min_count: int
            Threshold under which the words won't be taken into account to update the vocabulary.
            If a word is not in the vocabulary and has a frequency < min_count, the token will be considered as <UNKNOWN> : 3.
            
        """

        ## Model Definition
        def flatten(l):
            return list(itertools.chain.from_iterable(l))

        frequency = Counter(flatten(corpus))
        vocabulary_corpus = set([x for x in list(frequency.keys()) if frequency[x] >= min_count]) 
        if self.fitted:

            print("Loading Model...")
            new_vocabulary = vocabulary_corpus - self.vocabulary
            vocab_size = len(self.vocabulary)
            add_to_dic_w2i = dict(zip(new_vocabulary, range(vocab_size + self.n_special_tokens, vocab_size + len(new_vocabulary) + self.n_special_tokens)))
            add_to_dic_i2w = dict(zip(range(vocab_size + self.n_special_tokens, vocab_size + len(new_vocabulary) + self.n_special_tokens), new_vocabulary))
            self.word2idx.update(add_to_dic_w2i) 
            self.idx2word.update(add_to_dic_i2w) 
            vocabulary = self.vocabulary.union(vocabulary_corpus)
            assert len(vocabulary) == (self.vocab_size + len(new_vocabulary))
            self.vocabulary = vocabulary
            self.vocab_size = len(self.vocabulary)
            self.list_vocabulary = list(vocabulary)

            # Characters
            characters_corpus = set(Counter(''.join(list(self.vocabulary))).keys())
            new_characters = characters_corpus - set(self.char2idx.keys())
            n_char = len(self.char2idx)
            add_to_dic_c2i = dict(zip(new_characters, range(n_char + 1, n_char + len(new_characters) + 1)))
            self.char2idx.update(add_to_dic_c2i)

        else:

            print("Initializing Model")
            self.vocabulary = vocabulary_corpus 
            self.vocab_size = len(self.vocabulary)

            self.word2idx = {}
            self.idx2word = {}
            for i, word in enumerate(self.vocabulary):
                self.word2idx[word] = i + self.n_special_tokens
                self.idx2word[i + self.n_special_tokens] = word  
            self.list_vocabulary = list(self.vocabulary)

            characters_corpus = set(Counter(''.join(list(self.vocabulary))).keys())
            self.char2idx = {}
            for i, char in enumerate(characters_corpus):
                self.char2idx[char] = i + self.n_special_characters
            self.n_char = len(characters_corpus)

        ## Dataset
        source_text = []
        indexed_text = []
        for sentence in corpus:
          indexes = list(map(self.get_index_word, sentence))
          if len(sentence) >= min_size:
            source_text.append(sentence)
            indexed_text.append(indexes)
        source_text = np.array(source_text)
        indexed_text = np.array(indexed_text)

        ## Training
        print("Training...")
        self.fitted = True
        self.save_model()
        for _ in range(epochs):
            set_index = set(range(len(source_text)))
            progbar = tf.keras.utils.Progbar(len(source_text))
            iterations = 0
            while len(set_index) > 0:
                inp_words, inp_chars = self.get_next_batch(batch_size, set_index, source_text, indexed_text)
                generator_loss = self.train_step_generator(inp_words, inp_chars)
                progbar.add(inp_words.shape[0], values = [('Gen. Loss', generator_loss)])
                iterations += 1
                if (iterations % 5000) == 0 and STORAGE_BUCKET != None:
                    self.ckpt_manager.save()
                    self.save_model()
                    self.upload_to_cloud(STORAGE_BUCKET)
            self.ckpt_manager.save()


    def save_model(self):
        
        """
        Saves the model.
        
        """
        parameters = {
            'fitted' : self.fitted,
            'word2idx' : self.word2idx,
            'char2idx' : self.char2idx,
            'vocabulary' : list(self.vocabulary),
            'vocab_size' : self.vocab_size,
            'd_model' : self.d_model,
            'num_layers' : self.num_layers,
            'pe_input' : self.pe_input,
            'filters' : self.filters,
            'd_embeddings' : self.d_embeddings,
            'num_highway_layers' : self.num_highway_layers
        }
        
        with open(os.path.join(self.path_model, 'parameters.json'), 'w') as params:
            json.dump(parameters, params)

        return parameters

    def upload_to_cloud(self, STORAGE_BUCKET):
        command = "gsutil -m cp -r {} {}".format(os.path.join(self.path_model, "tf_ckpts"), STORAGE_BUCKET)
        os.system(command)

        command = "gsutil -m cp -r {} {}".format(os.path.join(self.path_model, "parameters.json"), STORAGE_BUCKET)
        os.system(command)
