import operator
from nltk import word_tokenize, sent_tokenize
from string import punctuation


def preprocess_words(raw_words):
    """ Eliminate punctuations and change to lowercase """
    punctuations = list(punctuation)
    punctuations.append('\'\'')
    punctuations.append('``')

    words = []

    for word in raw_words:
        if word not in punctuations:
            words.append(word.lower())

    return words


def split_sentences(text):
    return sent_tokenize(text)


def split_words(text):
    raw_words = word_tokenize(text)

    words = preprocess_words(raw_words)

    return words


def update_freq_dict(freq_dict, words):
    for word in words:
        if word in freq_dict:
            freq_dict[word] += 1
        else:
            freq_dict[word] = 1


def sort_dict_by_value(d):
    d_sorted = sorted(d.items(), key=operator.itemgetter(1))
    return d_sorted


def sort_dict_by_key(d):
    d_sorted = sorted(d.items(), key=operator.itemgetter(0))
    return d_sorted
