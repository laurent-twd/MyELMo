import os
import json

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa
import numpy as np
import random

from collections import Counter
import itertools

from encoder import BertEncoder
from optimizer.custom_schedule import CustomSchedule
from utilities.utils import create_padding_mask

MAX_VOCAB_SIZE = 125000
MAX_N_CHARS = 500

class MyELECTRA:

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
        self.dff = parameters['dff']
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


        self.generator = BertEncoder(vocab_size = MAX_VOCAB_SIZE,
                                    n_chars = MAX_N_CHARS,
                                    filters = self.filters,
                                    output_dim = MAX_VOCAB_SIZE,
                                    d_embeddings = self.d_embeddings,
                                    hidden_size = int(self.d_model / 3),
                                    num_layers = int(self.num_layers / 3),
                                    num_highway_layers = self.num_highway_layers,
                                    num_attention_heads = int(4 / 3),
                                    max_sequence_length = self.pe_input,
                                    inner_dim = self.dff,
                                    output_dropout = 0.1,
                                    attention_dropout = 0.1)

        # Model
        self.discriminator = BertEncoder(vocab_size = MAX_VOCAB_SIZE,
                                    n_chars = MAX_N_CHARS,
                                    filters = self.filters,
                                    output_dim = 1,
                                    d_embeddings = self.d_embeddings,
                                    hidden_size = self.d_model,
                                    num_layers = self.num_layers,
                                    num_highway_layers = self.num_highway_layers,
                                    num_attention_heads = 4 ,
                                    max_sequence_length = self.pe_input,
                                    inner_dim = self.dff,
                                    output_dropout = 0.1,
                                    attention_dropout = 0.1)

        # Optimizer

        gen_learning_rate = CustomSchedule()
        disc_learning_rate = CustomSchedule()
        self.generator_optimizer = tfa.optimizers.AdamW(
                                                weight_decay = 5e-4,
                                                learning_rate = gen_learning_rate,
                                                beta_1 = 0.9,
                                                beta_2 = 0.999,
                                                epsilon = 1e-06)

        self.discriminator_optimizer = tfa.optimizers.AdamW(
                                                weight_decay = 5e-4,
                                                learning_rate = disc_learning_rate,
                                                beta_1 = 0.9,
                                                beta_2 = 0.999,
                                                epsilon = 1e-06)

        self.gen_decay_var_list	= [v for v in self.generator.variables[:-2] if 'layer_norm' not in v.name and 'bias' not in v.name]
        self.disc_decay_var_list = [v for v in self.discriminator.variables[:-2] if 'layer_norm' not in v.name and 'bias' not in v.name]

        self.ckpt = tf.train.Checkpoint(generator = self.generator, 
                                        discriminator = self.discriminator,
                                        generator_optimizer = self.generator_optimizer,
                                        discriminator_optimizer = self.discriminator_optimizer)
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

    def process_sentence(self, tar_sentence, tar_indexes, masking_rate):
        
        inp_sentence = ['[CLS]']
        inp_indexes = [1]
        new_tar_indexes = [1]
        length = len(tar_sentence)
        number_of_words_masked = int(masking_rate * length)
        admissible_indexes = np.where(np.array(tar_indexes) != 3)[0]
        number_of_words_masked = np.minimum(int(masking_rate * length), len(admissible_indexes))
        masked_indexes = np.sort(np.random.choice(admissible_indexes, number_of_words_masked, replace = False))
        type_mask = np.random.choice([0, 1, 2], size = number_of_words_masked, replace = True, p = [0.8, 0.1, 0.1])
        j = 0
        for i in range(len(tar_sentence)):
            if j < len(masked_indexes) and i==masked_indexes[j]:
                if type_mask[j]==0:
                    inp_sentence.append('[MASKED]')
                    inp_indexes.append(4)
                elif type_mask[j]==1:
                    temp = np.random.choice(self.list_vocabulary)
                    inp_sentence.append(temp)
                    inp_indexes.append(self.get_index_word(temp))
                else:
                    inp_sentence.append(tar_sentence[i])
                    inp_indexes.append(tar_indexes[i])     
                j+=1
            else:
                inp_sentence.append(tar_sentence[i])
                inp_indexes.append(tar_indexes[i])
            new_tar_indexes.append(tar_indexes[i])

        inp_sentence.append('[SEP]')
        inp_indexes.append(2)
        new_tar_indexes.append(2)

        return inp_sentence, inp_indexes, new_tar_indexes, masked_indexes + 1
        
    def get_next_batch(self, batch_size, set_index, source_text, indexed_text, masking_rate):

        num_samples = np.minimum(batch_size, len(set_index))
        target_indexes = random.sample(set_index, num_samples)
        set_index.difference_update(set(target_indexes))

        tar_text = source_text[target_indexes]
        tar_indexes = indexed_text[target_indexes]
        temp = list(map(lambda x: self.process_sentence(tar_text[x], tar_indexes[x], masking_rate), range(num_samples)))
        inp_text, inp_indexes, new_tar_indexes, masked_idx = list(zip(*temp))

        max_len_char = 25 
        max_len = self.pe_input#min(max(list(map(len, inp_text))), self.pe_input)

        language_mask = tf.concat(list(map(lambda x: tf.reduce_sum(tf.one_hot(x, depth = max_len), axis = 0)[tf.newaxis, :], list(masked_idx))), axis = 0)
        inp_chars = list(map(lambda x: self.pad_char(x, max_len, max_len_char)[tf.newaxis, :, :], inp_text))
        inp_chars = tf.concat(inp_chars, axis = 0)

        inp_words = tf.cast(tf.keras.preprocessing.sequence.pad_sequences(inp_indexes, maxlen = max_len, padding = 'post'), tf.int32)
        tar_words = tf.cast(tf.keras.preprocessing.sequence.pad_sequences(new_tar_indexes, maxlen = max_len, padding = 'post'), tf.int32)

        return inp_words, inp_chars, tar_words, language_mask

    def get_input_discriminator(self, inp_chars, tar_words, gen_words, language_mask):

        get_text = np.vectorize(lambda x: self.get_word_index(x))
        gen_text = list(get_text(gen_words))
        max_len_char = 25 
        max_len = self.pe_input#min(max(list(map(len, inp_text))), self.pe_input)

        gen_chars = list(map(lambda x: self.pad_char(x, max_len, max_len_char)[tf.newaxis, :, :], gen_text))
        gen_chars = tf.concat(gen_chars, axis = 0)

        char_language_mask = tf.tile(language_mask[:, :, tf.newaxis], [1, 1, max_len_char])
        char_language_mask = tf.cast(char_language_mask, dtype = tf.bool)
        gen_chars = tf.where(char_language_mask, gen_chars, inp_chars)

        gen_is_true = tf.cast(tar_words == gen_words, dtype = tf.float32)
        adversarial_mask = tf.maximum(0., language_mask - gen_is_true)

        return gen_words, gen_chars, adversarial_mask

    @tf.function
    def train_step_generator(self, inp_words, inp_chars, tar_words, language_mask):

        enc_padding_mask = create_padding_mask(inp_words)

        with tf.GradientTape() as tape:
            gen_logits = self.generator([inp_chars, enc_padding_mask], training = True)['logits']

            mask_logits = tf.concat([tf.zeros(self.n_special_tokens + self.vocab_size), tf.ones(MAX_VOCAB_SIZE  - self.n_special_tokens - self.vocab_size)], 0)
            gen_logits += mask_logits[tf.newaxis, tf.newaxis, :] * (-1e9)
            loss = tf.keras.losses.sparse_categorical_crossentropy(tar_words, gen_logits, from_logits = True)
            mask = language_mask 
            loss = tf.math.divide_no_nan(tf.reduce_sum(loss * mask, axis = 1), tf.reduce_sum(mask, axis = 1))
            batch_loss = tf.reduce_mean(loss)

            variables = self.generator.trainable_variables[:-2]
            gradients = tape.gradient(batch_loss, variables)    
            self.generator_optimizer.apply_gradients(zip(gradients, variables), decay_var_list = self.gen_decay_var_list)
        
        return batch_loss, gen_logits, enc_padding_mask
               
    @tf.function
    def train_step_discriminator(self, gen_words, gen_chars, enc_padding_mask, adversarial_mask):

        with tf.GradientTape() as tape:
            disc_logits = self.discriminator([gen_chars, enc_padding_mask], training = True)['logits']
            probs = tf.squeeze(tf.math.sigmoid(disc_logits + 1e-9), axis = 2)
            loss = adversarial_mask * tf.math.log(probs) + (1 - adversarial_mask) * tf.math.log(1. - probs)
            padding_mask = 1. - enc_padding_mask
            mask = padding_mask
            loss = tf.math.divide_no_nan(tf.reduce_sum(loss * mask, axis = 1), tf.reduce_sum(mask, axis = 1))
            batch_loss = - tf.reduce_mean(loss)

            variables = self.discriminator.trainable_variables[:-2]
            gradients = tape.gradient(batch_loss, variables)    
            self.discriminator_optimizer.apply_gradients(zip(gradients, variables), decay_var_list = self.disc_decay_var_list)
        
        return batch_loss

    def get_gen_words(self, gen_logits, num_splits = 1):
        if num_splits > 1:
            gen_words = []
            split_gen_logits = tf.split(gen_logits, num_or_size_splits = num_splits, axis = 0)
            for split in split_gen_logits:
                gen_words.append(tfp.distributions.Categorical(logits = split).sample())
            gen_words = tf.concat(gen_words, axis = 0)
        else:
            gen_words = tfp.distributions.Categorical(logits = gen_logits).sample()
            
        return gen_words

    def fit(self, corpus, epochs, batch_size, masking_rate = 0.15, min_count = 1, STORAGE_BUCKET = None):
        
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
        limit = int(1 / masking_rate) + 1
        for sentence in corpus:
          indexes = list(map(self.get_index_word, sentence))
          if len(sentence) >= limit:
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
                inp_words, inp_chars, tar_words, language_mask = self.get_next_batch(batch_size, set_index, source_text, indexed_text, masking_rate)
                generator_loss, gen_logits, enc_padding_mask = self.train_step_generator(inp_words, inp_chars, tar_words, language_mask)
                gen_words = self.get_gen_words(gen_logits, num_splits = 1)
                gen_words, gen_chars, adversarial_mask = self.get_input_discriminator(inp_chars, tar_words, gen_words, language_mask)
                discriminator_loss = self.train_step_discriminator(gen_words, gen_chars, enc_padding_mask, adversarial_mask)
                progbar.add(inp_words.shape[0], values = [('Gen. Loss', generator_loss), ('Disc. Loss', discriminator_loss)])
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
            'dff' : self.dff,
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
