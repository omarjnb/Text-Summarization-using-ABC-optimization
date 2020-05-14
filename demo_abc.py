import nltk
import os
import re
import math
import operator
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.stem.porter import PorterStemmer
import numpy as np
from objective_func import content_coverage
from abc_ import ABC

nltk.download('averaged_perceptron_tagger')
Stopwords = set(stopwords.words('english'))
wordlemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def lemmatize_words(words):
    lemmatized_words = []
    for word in words:
       lemmatized_words.append(wordlemmatizer.lemmatize(word))
    return lemmatized_words

def stem_words(words):
    stemmed_words = []
    for word in words:
       stemmed_words.append(stemmer.stem(word))
    return stemmed_words

def remove_special_characters(text):
    regex = r'[^a-zA-Z0-9\s]'
    text = re.sub(regex,'',text)
    return text

def freq(words):
    '''returns a dict of frequency of words in document'''
    words = [word.lower() for word in words]
    dict_freq = {}
    words_unique = []
    for word in words:
       if word not in words_unique:
           words_unique.append(word)
    for word in words_unique:
       dict_freq[word] = words.count(word)
    return dict_freq

def pos_tagging(text):
    pos_tag = nltk.pos_tag(text.split())
    pos_tagged_noun_verb = []
    for word,tag in pos_tag:
        if tag == "NN" or tag == "NNP" or tag == "NNS" or tag == "VB" or tag == "VBD" or tag == "VBG" or tag == "VBN" or tag == "VBP" or tag == "VBZ":
             pos_tagged_noun_verb.append(word)
    return pos_tagged_noun_verb

def tf_score(word,sentence):
    word_frequency_in_sentence = 0
    len_sentence = len(sentence)
    for word_in_sentence in sentence.split():
        if word == word_in_sentence:
            word_frequency_in_sentence = word_frequency_in_sentence + 1
    tf =  word_frequency_in_sentence/ len_sentence
    return tf

def idf_score(no_of_sentences,word,sentences):
    no_of_sentence_containing_word = 0
    for sentence in sentences:
        sentence = remove_special_characters(str(sentence))
        sentence = re.sub(r'\d+', '', sentence)
        sentence = sentence.split()
        sentence = [word for word in sentence if word.lower() not in Stopwords and len(word)>1]
        sentence = [word.lower() for word in sentence]
        sentence = [wordlemmatizer.lemmatize(word) for word in sentence]
        if word in sentence:
            no_of_sentence_containing_word = no_of_sentence_containing_word + 1
    if(no_of_sentence_containing_word == 0):
        no_of_sentence_containing_word = 1
    idf = math.log10(no_of_sentences/no_of_sentence_containing_word)
    return idf

def tf_idf_score(tf,idf):
    return tf*idf

def word_tfidf(dict_freq,word,sentences,sentence):
    '''tf_idf score of word in the sentence'''
    tf = tf_score(word,sentence)
    idf = idf_score(len(sentences),word,sentences)
    tf_idf = tf_idf_score(tf,idf)
    return tf_idf

def sentence_vector(sentence,dict_freq,sentences):
    '''returns a map of tf_idf scores of word, where word is key and score is value. all unique words have a score in the map.'''
    # dict_freq has the frequency of every word in the document
    sentence_score = dict()
    sentence = remove_special_characters(str(sentence)) 
    sentence = re.sub(r'\d+', '', sentence)
    for word in dict_freq.keys():
        sentence_score[word] = word_tfidf(dict_freq, word, sentences, sentence)
    return sentence_score

def mean_vector(sentence_score,no_of_sentences, dict_freq):
    mean = dict()
    for word in dict_freq:
        su = 0
        for i in range(no_of_sentences):
            su += sentences_with_score[i][word]
        mean[word] = su/no_of_sentences
    return mean

def print_pos(pos, sentences):
    summary=""
    for i in range(len(pos)):
        if(pos[i] == 1):
            summary += sentences[i]
    print(summary)

def simulate(obj_function, sentences, colony_size=30, n_iter=5000, max_trials=100, simulations=30):
    # for _ in range(simulations):
        # optimizer = ABC(obj_function, colony_size=colony_size, n_iter=n_iter, max_trials=max_trials)
        # optimizer.optimize()
    optimizer = ABC(obj_function, colony_size=colony_size, n_iter=n_iter, max_trials=max_trials)
    optimizer.optimize()
    print_pos(optimizer.optimal_solution.pos, sentences)

file = 'input.txt'
file = open(file , 'r')
text = file.read()
tokenized_sentence = sent_tokenize(text)
text = remove_special_characters(str(text))
text = re.sub(r'\d+', '', text)
tokenized_words_with_stopwords = word_tokenize(text)
tokenized_words = [word for word in tokenized_words_with_stopwords if word not in Stopwords]
tokenized_words = [word for word in tokenized_words if len(word) > 1]
tokenized_words = [word.lower() for word in tokenized_words]
tokenized_words = lemmatize_words(tokenized_words)
word_freq = freq(tokenized_words)
input_user = int(input('Percentage of information to retain(in percent):'))
no_of_sentences = int((input_user * len(tokenized_sentence))/100)
sentences_with_score = dict()
for i in range(len(tokenized_sentence)):
    sentences_with_score[i] = sentence_vector(tokenized_sentence[i], word_freq, tokenized_sentence)
o = mean_vector(sentences_with_score, len(tokenized_sentence), word_freq)
obj_function = content_coverage(len(tokenized_sentence), no_of_sentences, o, sentence_map=sentences_with_score)
# for x in sentences_with_score.keys():
#     print(x, sentences_with_score[x])
simulate(obj_function, tokenized_sentence)