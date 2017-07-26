# -*- coding: utf-8 -*-

# NOTE: this is NOT tensorflow. This is PyTorch implementation, standalone of GAN.

"""
Translation with a Sequence to Sequence Network and Attention
*************************************************************
**Author**: `Sean Robertson <https://github.com/spro/practical-pytorch>`_

In this project we will be teaching a neural network to translate from
French to English.

::

    [KEY: > input, = target, < output]

    > il est en train de peindre un tableau .
    = he is painting a picture .
    < he is painting a picture .

    > pourquoi ne pas essayer ce vin delicieux ?
    = why not try that delicious wine ?
    < why not try that delicious wine ?

    > elle n est pas poete mais romanciere .
    = she is not a poet but a novelist .
    < she not not a poet but a novelist .

    > vous etes trop maigre .
    = you re too skinny .
    < you re all alone .

... to varying degrees of success.

This is made possible by the simple but powerful idea of the `sequence
to sequence network <http://arxiv.org/abs/1409.3215>`__, in which two
recurrent neural networks work together to transform one sequence to
another. An encoder network condenses an input sequence into a vector,
and a decoder network unfolds that vector into a new sequence.

.. figure:: /_static/img/seq-seq-images/seq2seq.png
   :alt: 

To improve upon this model we'll use an `attention
mechanism <https://arxiv.org/abs/1409.0473>`__, which lets the decoder
learn to focus over a specific range of the input sequence.

**Recommended Reading:**

I assume you have at least installed PyTorch, know Python, and
understand Tensors:

-  http://pytorch.org/ For installation instructions
-  :doc:`/beginner/deep_learning_60min_blitz` to get started with PyTorch in general
-  :doc:`/beginner/pytorch_with_examples` for a wide and deep overview
-  :doc:`/beginner/former_torchies_tutorial` if you are former Lua Torch user


It would also be useful to know about Sequence to Sequence networks and
how they work:

-  `Learning Phrase Representations using RNN Encoder-Decoder for
   Statistical Machine Translation <http://arxiv.org/abs/1406.1078>`__
-  `Sequence to Sequence Learning with Neural
   Networks <http://arxiv.org/abs/1409.3215>`__
-  `Neural Machine Translation by Jointly Learning to Align and
   Translate <https://arxiv.org/abs/1409.0473>`__
-  `A Neural Conversational Model <http://arxiv.org/abs/1506.05869>`__

You will also find the previous tutorials on
:doc:`/intermediate/char_rnn_classification_tutorial`
and :doc:`/intermediate/char_rnn_generation_tutorial`
helpful as those concepts are very similar to the Encoder and Decoder
models, respectively.

And for more, read the papers that introduced these topics:

-  `Learning Phrase Representations using RNN Encoder-Decoder for
   Statistical Machine Translation <http://arxiv.org/abs/1406.1078>`__
-  `Sequence to Sequence Learning with Neural
   Networks <http://arxiv.org/abs/1409.3215>`__
-  `Neural Machine Translation by Jointly Learning to Align and
   Translate <https://arxiv.org/abs/1409.0473>`__
-  `A Neural Conversational Model <http://arxiv.org/abs/1506.05869>`__

**Requirements**
"""



from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import spacy
from spacy.en import English
spacynlp = English()

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

# MODIFIED from original code
# the following imports are for reading SQuAD json files
import nltk
import json
import numpy as np
import os

use_cuda = torch.cuda.is_available()

######################################################################
# Loading data files
# ==================
#
# The data for this project is a set of many thousands of English to
# French translation pairs.
#
# `This question on Open Data Stack
# Exchange <http://opendata.stackexchange.com/questions/3888/dataset-of-sentences-translated-into-many-languages>`__
# pointed me to the open translation site http://tatoeba.org/ which has
# downloads available at http://tatoeba.org/eng/downloads - and better
# yet, someone did the extra work of splitting language pairs into
# individual text files here: http://www.manythings.org/anki/
#
# The English to French pairs are too big to include in the repo, so
# download to ``data/eng-fra.txt`` before continuing. The file is a tab
# separated list of translation pairs:
#
# ::
#
#     I am cold.    Je suis froid.
#
# .. Note::
#    Download the data from
#    `here <https://download.pytorch.org/tutorial/data.zip>`_
#    and extract it to the current directory.

######################################################################
# Similar to the character encoding used in the character-level RNN
# tutorials, we will be representing each word in a language as a one-hot
# vector, or giant vector of zeros except for a single one (at the index
# of the word). Compared to the dozens of characters that might exist in a
# language, there are many many more words, so the encoding vector is much
# larger. We will however cheat a bit and trim the data to only use a few
# thousand words per language.
#
# .. figure:: /_static/img/seq-seq-images/word-encoding.png
#    :alt:
#
#


######################################################################
# We'll need a unique index per word to use as the inputs and targets of
# the networks later. To keep track of all this we will use a helper class
# called ``Lang`` which has word → index (``word2index``) and index → word
# (``index2word``) dictionaries, as well as a count of each word
# ``word2count`` to use to later replace rare words.
#


# # this class now will need to find the mapping from word to its vector through the embedding_index dictionary
# class Lang:
#     def __init__(self, name):
#         self.name = name
#         self.word2index = {}
#         self.word2count = {}
#         self.index2word = {0: "SOS", 1: "EOS"}
#         self.n_words = 2  # Count SOS and EOS

#     def addSentence(self, sentence):
#         for word in sentence.split(' '):
#             self.addWord(word)

#     def addWord(self, word):
#         if word not in self.word2index:
#             self.word2index[word] = self.n_words
#             self.word2count[word] = 1
#             self.index2word[self.n_words] = word
#             self.n_words += 1
#         else:
#             self.word2count[word] += 1


######################################################################
# The files are all in Unicode, to simplify we will turn Unicode
# characters to ASCII, make everything lowercase
#

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    # s = re.sub(r"([.!?])", r" \1", s)
    # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


######################################################################
# MODIFIED from original code
# read data specific for SQUAD dataset

def readSQuAD(path_to_data):
    # output (context, question, answer) triplets
    print("Reading dataset...")
    triplets = []
    with open(path_to_data) as f:
        train = json.load(f)
        train = train['data']
        for s in range(0, len(train)):
            samples = train[s]['paragraphs']
            for p in range(0, len(samples)):
                context = samples[p]['context']
                # turn from unicode to ascii and lower case everything
                context = normalizeString(context)
                qas = samples[p]['qas']
                for i in range(0, len(qas)):
                # print('current s,p,i are: ' + str(s)+str(p)+str(i))
                    answers = qas[i]['answers']
                    question = qas[i]['question']
                    # turn from unicode to ascii and lower case everything
                    question = normalizeString(question)
                    for a in range(0, len(answers)):
                        ans_text = answers[a]['text']
                        # turn from unicode to ascii and lower case everything
                        ans_text = normalizeString(ans_text)
                        triplets.append((context, question, ans_text))
    # c_lang = Lang(lang1)
	# q_lang = Lang(lang2)
	# a_lang = Lang(lang3)
	# all_lang = Lang(lang4)
    return triplets


######################################################################
# Since there are a *lot* of example sentences and we want to train
# something quickly, we'll trim the data set to only relatively short and
# simple sentences. Here the maximum length is 10 words (that includes
# ending punctuation) and we're filtering to sentences that translate to
# the form "I am" or "He is" etc. (accounting for apostrophes replaced
# earlier).
#

# MODIFIED: in our case we will NOT do the filtering
# longest sequence in context in dataset is 653, so just set max_len to be 1000

# MAX_LENGTH = 1000

# # eng_prefixes = (
# #     "i am ", "i m ",
# #     "he is", "he s ",
# #     "she is", "she s",
# #     "you are", "you re ",
# #     "we are", "we re ",
# #     "they are", "they re "
# # )


# def filterTriple(p):
#     return len(p[0].split(' ')) < MAX_LENGTH and \
#         len(p[1].split(' ')) < MAX_LENGTH #and \
#         # p[1].startswith(eng_prefixes)


# def filterTriplets(triplets):
#     return [triple for triple in triplets if filterTriple(triple)]


######################################################################
# The full process for preparing the data is:
#
# -  (Read dataset) <-- MODIFIED
# -  Normalize text, (filter by length and content) <-- MODIFIED: no this step
# -  Make word lists from sentences in pairs
#

# def prepareData(path_to_data, lang1, lang2, lang3, lang4):
#     c_lang, q_lang, a_lang, all_lang, triplets = readLangs(path_to_data, lang1, lang2, lang3, lang4)
#     print("Read %s (context, question, answer) triplets" % len(triplets))
#     # MODIFIED: commented out the following lines because we DO NOT do any filtering
#     # pairs = filterPairs(pairs)
#     # print("Trimmed to %s sentence pairs" % len(pairs))
#     print("Counting words...")
#     for triple in triplets:
#         c_lang.addSentence(triple[0]) # this is more a paragraph (multiple sentences) rather than a single sentence
#         q_lang.addSentence(triple[1])
#         a_lang.addSentence(triple[2])
#         all_lang.addSentence(triple[0] + ' ' +triple[2]) # only add context and question together because answer is a subset of context
#     print("Counted words in dataset (all questions + all contexts):")
#     print(all_lang.name, all_lang.n_words)
#     return c_lang, q_lang, a_lang, all_lang, triplets

# c_lang, q_lang, a_lang, all_lang, triplets = prepareData(path_to_data, 'context', 'question', 'answer', 'all')
# print(random.choice(triplets))


#
# Preparing Training Data
# -----------------------
#
# To train, for each pair we will need an input tensor (indexes of the
# words in the input sentence) and target tensor (indexes of the words in
# the target sentence). While creating these vectors we will append the
# EOS token to both sequences.
#


def tokenizeSentence(sentence, embeddings_index, embeddings_size):
    tokenized_sentence = spacynlp.tokenizer(sentence)
    token_num = len(tokenized_sentence)
    var = torch.FloatTensor(token_num+1, embeddings_size) #add one dimension for EOS
    # var[0] = embeddings_index['SOS']
    for t in range(0, token_num):
        try:
            var[t] = embeddings_index[tokenized_sentence[t].string.strip()]
        except KeyError:
            print('original word>')
            print(tokenized_sentence[t])
            print('string format>')
            print(tokenized_sentence[t].string)
            print(sentence)
            print('-------------------------------------')
            print('-------------------------------------')
    # add end of sentence token to all sentences
    var[-1] = embeddings_index['EOS']
    return var


# def variableFromSentence(sentence, embeddings_index):
#     # indexes = indexesFromSentence(lang, sentence)
#     # indexes.append(EOS_token)
#     result = Variable(torch.LongTensor(indexes).view(-1, 1))
#     if use_cuda:
#         return result.cuda()
#     else:
#         return result


def variablesFromTriplets(triple, embeddings_index, embeddings_size):
    context = tokenizeSentence(triple[0], embeddings_index, embeddings_size)
    answer = tokenizeSentence(triple[2], embeddings_index, embeddings_size)
    question = tokenizeSentence(triple[1], embeddings_index, embeddings_size)
    return (Variable(context), Variable(question), Variable(answer))




######################################################################
# The Seq2Seq Model
# =================
#
# A Recurrent Neural Network, or RNN, is a network that operates on a
# sequence and uses its own output as input for subsequent steps.
#
# A `Sequence to Sequence network <http://arxiv.org/abs/1409.3215>`__, or
# seq2seq network, or `Encoder Decoder
# network <https://arxiv.org/pdf/1406.1078v3.pdf>`__, is a model
# consisting of two RNNs called the encoder and decoder. The encoder reads
# an input sequence and outputs a single vector, and the decoder reads
# that vector to produce an output sequence.
#
# .. figure:: /_static/img/seq-seq-images/seq2seq.png
#    :alt:
#
# Unlike sequence prediction with a single RNN, where every input
# corresponds to an output, the seq2seq model frees us from sequence
# length and order, which makes it ideal for translation between two
# languages.
#
# Consider the sentence "Je ne suis pas le chat noir" → "I am not the
# black cat". Most of the words in the input sentence have a direct
# translation in the output sentence, but are in slightly different
# orders, e.g. "chat noir" and "black cat". Because of the "ne/pas"
# construction there is also one more word in the input sentence. It would
# be difficult to produce a correct translation directly from the sequence
# of input words.
#
# With a seq2seq model the encoder creates a single vector which, in the
# ideal case, encodes the "meaning" of the input sequence into a single
# vector — a single point in some N dimensional space of sentences.
#


######################################################################
# The Encoder
# -----------
#
# The encoder of a seq2seq network is a RNN that outputs some value for
# every word from the input sentence. For every input word the encoder
# outputs a vector and a hidden state, and uses the hidden state for the
# next input word.
#
# .. figure:: /_static/img/seq-seq-images/encoder-network.png
#    :alt:
#
#

class EncoderRNN(nn.Module):
	# output is the same dimension as input (dimension defined by externalword embedding model)
    def __init__(self, input_size, hidden_size, embeddings_index, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.embeddings_index = embeddings_index

        # self.embedding = nn.Embedding(input_size, input_dim)
        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embeddings_index[input].view(1, 1, -1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

######################################################################
# The Decoder
# -----------
#
# The decoder is another RNN that takes the encoder output vector(s) and
# outputs a sequence of words to create the translation.
#


######################################################################
# Simple Decoder
# ^^^^^^^^^^^^^^
#
# In the simplest seq2seq decoder we use only last output of the encoder.
# This last output is sometimes called the *context vector* as it encodes
# context from the entire sequence. This context vector is used as the
# initial hidden state of the decoder.
#
# At every step of decoding, the decoder is given an input token and
# hidden state. The initial input token is the start-of-string ``<SOS>``
# token, and the first hidden state is the context vector (the encoder's
# last hidden state).
#
# .. figure:: /_static/img/seq-seq-images/decoder-network.png
#    :alt:
#
#

class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        # TODO this following embedding should change. 
        # each embedding is of dimension input_dim defined by external word embedding
        # self.embedding = nn.Embedding(input_size, input_dim)
        self.gru = nn.GRU(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        output = embeddings_index[input].view(1, 1, -1)
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

######################################################################
# I encourage you to train and observe the results of this model, but to
# save space we'll be going straight for the gold and introducing the
# Attention Mechanism.
#


######################################################################
# Attention Decoder
# ^^^^^^^^^^^^^^^^^
#
# If only the context vector is passed betweeen the encoder and decoder,
# that single vector carries the burden of encoding the entire sentence.
#
# Attention allows the decoder network to "focus" on a different part of
# the encoder's outputs for every step of the decoder's own outputs. First
# we calculate a set of *attention weights*. These will be multiplied by
# the encoder output vectors to create a weighted combination. The result
# (called ``attn_applied`` in the code) should contain information about
# that specific part of the input sequence, and thus help the decoder
# choose the right output words.
#
# .. figure:: https://i.imgur.com/1152PYf.png
#    :alt:
#
# Calculating the attention weights is done with another feed-forward
# layer ``attn``, using the decoder's input and hidden state as inputs.
# Because there are sentences of all sizes in the training data, to
# actually create and train this layer we have to choose a maximum
# sentence length (input length, for encoder outputs) that it can apply
# to. Sentences of the maximum length will use all the attention weights,
# while shorter sentences will only use the first few.
#
# .. figure:: /_static/img/seq-seq-images/attention-decoder-network.png
#    :alt:
#
#

class AttnDecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, 
        embeddings_index, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.input_size = input_size
        # self.enc_output_len = enc_output_len
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.embeddings_index = embeddings_index
        # self.max_length = max_length

        # self.embedding = nn.Embedding(self.output_size, self.input_dim)
        # self.attn = nn.Linear(self.input_size+self.hidden_size, self.enc_output_len)
        self.attn_combine = nn.Linear(self.input_size+self.hidden_size, self.input_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.input_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_output, encoder_outputs):

        # because the number of input tokens varies, we move the init of attn to here
        # instead of in __init__ function
        attn = nn.Linear(self.input_size+self.hidden_size, encoder_outputs.size()[0])

        embedded = self.embeddings_index[input].view(1, 1, -1)
        # embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            attn(torch.cat((embedded[0], hidden[0]), 1)))
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


######################################################################
# .. note:: There are other forms of attention that work around the length
#   limitation by using a relative position approach. Read about "local
#   attention" in `Effective Approaches to Attention-based Neural Machine
#   Translation <https://arxiv.org/abs/1508.04025>`__.
#
# Training
# ========



######################################################################
# Training the Model
# ------------------
#
# To train we run the input sentence through the encoder, and keep track
# of every output and the latest hidden state. Then the decoder is given
# the ``<SOS>`` token as its first input, and the last hidden state of the
# encoder as its first hidden state.
#
# "Teacher forcing" is the concept of using the real target outputs as
# each next input, instead of using the decoder's guess as the next input.
# Using teacher forcing causes it to converge faster but `when the trained
# network is exploited, it may exhibit
# instability <http://minds.jacobs-university.de/sites/default/files/uploads/papers/ESNTutorialRev.pdf>`__.
#
# You can observe outputs of teacher-forced networks that read with
# coherent grammar but wander far from the correct translation -
# intuitively it has learned to represent the output grammar and can "pick
# up" the meaning once the teacher tells it the first few words, but it
# has not properly learned how to create the sentence from the translation
# in the first place.
#
# Because of the freedom PyTorch's autograd gives us, we can randomly
# choose to use teacher forcing or not with a simple if statement. Turn
# ``teacher_forcing_ratio`` up to use more of it.
#

teacher_forcing_ratio = 0.5

# context = input_variable
def train(context_var, ans_var, question_var, embeddings_index,
    encoder1, encoder2, decoder, encoder_optimizer1, encoder_optimizer2, 
    decoder_optimizer, criterion):
    encoder_hidden_context = encoder1.initHidden()
    encoder_hidden_answer = encoder2.initHidden()
    decoder_hidden = decoder.initHidden()

    encoder_optimizer1.zero_grad()
    encoder_optimizer2.zero_grad()
    decoder_optimizer.zero_grad()

    input_length_context = context_var.size()[0]
    input_length_answer = ans_var.size()[0]
    target_length = question_var.size()[0]
    
    encoder_outputs_context = Variable(torch.zeros(input_length_context, encoder1.hidden_size))
    encoder_outputs_context = encoder_outputs_context.cuda() if use_cuda else encoder_outputs_context

    encoder_outputs_answer = Variable(torch.zeros(input_length_answer, encoder2.hidden_size))
    encoder_outputs_answer = encoder_outputs_answer.cuda() if use_cuda else encoder_outputs_answer
   
    loss = 0

    # context encoding
    for ei in range(input_length_context):
    	encoder_output_context, encoder_hidden_context = encoder1(
        	context_var[ei], encoder_hidden_context)
    	encoder_outputs_context[ei] = encoder_output_context[0][0]

    # answer encoding
    for ei in range(input_length_answer):
        encoder_output_answer, encoder_hidden_answer = encoder2(
            ans_var[ei], encoder_hidden_answer)
        encoder_outputs_answer[ei] = encoder_output_answer[0][0]

    # concat the context encoding and answer encoding
    encoder_output = torch.cat(encoder_output_context, encoder_output_answer)
    encoder_outputs = torch.cat(encoder_outputs_context, encoder_outputs_answer)

    decoder_input = Variable(embeddings_index['SOS'])
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    
    # decoder_hidden = torch.cat(encoder_hidden_context, encoder_hidden_answer)

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            loss += criterion(decoder_output[0], word2index(question_var[di]))
            decoder_input = question_var[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            
            decoder_input = Variable(embeddings_index(ni))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            
            loss += criterion(decoder_output[0], word2index(question_var[di]))
            if ni == EOS_token:
                break

    loss.backward()

    encoder_optimizer1.step()
    encoder_optimizer2.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
#

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


######################################################################
# The whole training process looks like this:
#
# -  Start a timer
# -  Initialize optimizers and criterion
# -  Create set of training pairs
# -  Start empty losses array for plotting
#
# Then we call ``train`` many times and occasionally print the progress (%
# of examples, time so far, estimated time) and average loss.
#

def trainIters(encoder1, encoder2, decoder, embeddings_index, 
    n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer1 = optim.SGD(encoder1.parameters(), lr=learning_rate)
    encoder_optimizer2 = optim.SGD(encoder2.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_triplets = [variablesFromTriplets(random.choice(triplets), 
                                                embeddings_index, embeddings_size)
                        for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_triple = training_triplets[iter - 1]
        context_var = training_triple[0]
        ans_var = training_triple[2]
        question_var = training_pair[1]
 
        loss = train(context_var, ans_var, question_var, encoder1, embeddings_index,
                     encoder2, decoder, encoder_optimizer1, encoder_optimizer2, 
                     decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            print('---sample generated question---')
            # sample a triple and print the generated question
            evaluateRandomly(encoder1, encoder2, decoder, triplets, n=1)
            print()

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


######################################################################
# Plotting results
# ----------------
#
# Plotting is done with matplotlib, using the array of loss values
# ``plot_losses`` saved while training.
#

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


######################################################################
# Evaluation
# ==========
#
# Evaluation is mostly the same as training, but there are no targets so
# we simply feed the decoder's predictions back to itself for each step.
# Every time it predicts a word we add it to the output string, and if it
# predicts the EOS token we stop there. We also store the decoder's
# attention outputs for display later.
#

def evaluate(encoder1, encoder2, decoder, triple):
    triple_var = variablesFromTriplets(triple, embeddings_index, embeddings_size)
    context_var = triple_var[0]
    ans_var = triple_var[2]
    input_length_context = context_var.size()[0]
    input_length_answer = ans_var.size()[0]
    encoder_hidden_context = encoder1.initHidden()
    encoder_hidden_answer = encoder2.initHidden()
    decoder_hidden = decoder.initHidden()


    encoder_outputs_context = Variable(torch.zeros(input_length_context, encoder1.hidden_size))
    encoder_outputs_context = encoder_outputs_context.cuda() if use_cuda else encoder_outputs_context
    encoder_outputs_answer = Variable(torch.zeros(input_length_answer, encoder2.hidden_size))
    encoder_outputs_answer = encoder_outputs_answer.cuda() if use_cuda else encoder_outputs_answer
   
    for ei in range(input_length_context):
        encoder_output_context, encoder_hidden_context = encoder1(context_var[ei],
                                                 encoder_hidden_context)
        encoder_outputs_context[ei] = encoder_outputs_context[ei] + encoder_output_context[0][0]

    for ei in range(input_length_answer):
        encoder_output_answer, encoder_hidden_answer = encoder2(ans_var[ei],
                                                 encoder_hidden_answer)
        encoder_outputs_answer[ei] = encoder_outputs_answer[ei] + encoder_output_answer[0][0]

    encoder_output = torch.cat(encoder_output_context, encoder_output_answer)
    encoder_outputs = torch.cat(encoder_outputs_context, encoder_outputs_answer)

    # decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = Variable(embeddings_index['SOS'])
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    # decoder_hidden = torch.cat(encoder_hidden_context, encoder_hidden_answer)

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, encoder_outputs.size()[0])

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_output, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(index2word[ni])
        
        decoder_input = Variable(embeddings(ni))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]


######################################################################
# We can evaluate random sentences from the training set and print out the
# input, target, and output to make some subjective quality judgements:
#

def evaluateRandomly(encoder1, encoder2, decoder, triplets, n=1):
    for i in range(n):
        triple = random.choice(triplets)
        print('context   > ', pair[0])
        print('question  > ', pair[1])
        print('answer    > ', pair[2])
        output_words, attentions = evaluate(encoder1, encoder2, decoder, triple)
        output_sentence = ' '.join(output_words)
        print('generated < ', output_sentence)
        print('')


######################################################################
# Training and Evaluating
# =======================
#
# With all these helper functions in place (it looks like extra work, but
# it's easier to run multiple experiments easier) we can actually
# initialize a network and start training.
#
# Remember that the input sentences were heavily filtered. For this small
# dataset we can use relatively small networks of 256 hidden nodes and a
# single GRU layer. After about 40 minutes on a MacBook CPU we'll get some
# reasonable results.
#
# .. Note:: 
#    If you run this notebook you can train, interrupt the kernel,
#    evaluate, and continue training later. Comment out the lines where the
#    encoder and decoder are initialized and run ``trainIters`` again.
#

######### set paths
# default values for the dataset and the path to the project/dataset
dataset = 'squad'
f_name = 'train-v1.1.json'
path_to_dataset = '/home/jack/Documents/QA_QG/data/'
path_to_data = path_to_dataset + dataset + '/' + f_name
GLOVE_DIR = path_to_dataset + 'glove.6B/'

######### first load the pretrained word embeddings
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    coefs = torch.from_numpy(coefs)
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# get dimension from a random sample in the dict
embeddings_size = random.sample( embeddings_index.items(), 1 )[0][1].size(-1)
SOS_token = -torch.ones(embeddings_size) # start of sentence token, all zerons
EOS_token = torch.ones(embeddings_size) # end of sentence token, all ones
# add special tokens to the embeddings
embeddings_index['SOS'] = SOS_token
embeddings_index['EOS'] = EOS_token

# read data
triplets = readSQuAD(path_to_data)

## find all unique tokens in the data (should be a subset of the number of embeddings)
data_tokens = ['SOS', 'EOS']
for triple in triplets:
    c = [str(token) for token in spacynlp.tokenizer(triple[0])]
    q = [str(token) for token in spacynlp.tokenizer(triple[1])]
    a = [str(token) for token in spacynlp.tokenizer(triple[2])]
    data_tokens += c + q + a
data_tokens = list(set(data_tokens)) # find unique

# build word2index dictionary and index2word dictionary
word2index = {}
index2word = {}
for i in range(0, len(data_tokens)):
    index2word[i] = data_tokens[i]
    word2index[data_tokens[i]] = i

print('reading and preprocessing data complete.')
print('found %s unique tokens in corpus.' % len(data_tokens))
if use_cuda:
    print('GPU ready.')
print('')
print('start training...')
print('')

hidden_size1 = 256
hidden_size2 = 64
# context encoder
encoder1 = EncoderRNN(embeddings_size, hidden_size1, embeddings_index)
# answer encoder
encoder2 = EncoderRNN(embeddings_size, hidden_size2, embeddings_index)
# decoder
attn_decoder1 = AttnDecoderRNN(embeddings_size, hidden_size1, embeddings_size, 
                                embeddings_index, 1, dropout_p=0.1)

if use_cuda:
    encoder1 = encoder1.cuda()
    encoder2 = encoder2.cuda()
    attn_decoder1 = attn_decoder1.cuda()

trainIters(encoder1, encoder2, attn_decoder1, embeddings_index, 75000, print_every=5000)

######################################################################
#

evaluateRandomly(encoder1, encoder2, attn_decoder1)


######################################################################
# Visualizing Attention
# ---------------------
#
# A useful property of the attention mechanism is its highly interpretable
# outputs. Because it is used to weight specific encoder outputs of the
# input sequence, we can imagine looking where the network is focused most
# at each time step.
#
# You could simply run ``plt.matshow(attentions)`` to see attention output
# displayed as a matrix, with the columns being input steps and rows being
# output steps:
#

output_words, attentions = evaluate(
    encoder1, attn_decoder1, "je suis trop froid .")
plt.matshow(attentions.numpy())


######################################################################
# For a better viewing experience we will do the extra work of adding axes
# and labels:
#

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)


evaluateAndShowAttention("elle a cinq ans de moins que moi .")

evaluateAndShowAttention("elle est trop petit .")

evaluateAndShowAttention("je ne crains pas de mourir .")

evaluateAndShowAttention("c est un jeune directeur plein de talent .")


######################################################################
# Exercises
# =========
#
# -  Try with a different dataset
#
#    -  Another language pair
#    -  Human → Machine (e.g. IOT commands)
#    -  Chat → Response
#    -  Question → Answer
#
# -  Replace the embeddings with pre-trained word embeddings such as word2vec or
#    GloVe
# -  Try with more layers, more hidden units, and more sentences. Compare
#    the training time and results.
# -  If you use a translation file where pairs have two of the same phrase
#    (``I am test \t I am test``), you can use this as an autoencoder. Try
#    this:
#
#    -  Train as an autoencoder
#    -  Save only the Encoder network
#    -  Train a new Decoder for translation from there
#
