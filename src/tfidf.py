import math
import csv
import os.path
import util
from nltk.corpus import brown


class Sentence:
    def __init__(self, text):
        self.text = text
        self.score = 0.0
        self.words = util.split_words(self.text)


class Tfidf:
    def __init__(self, words, sample_doc_ids):
        self.tf = {}
        self.idf = {}
        self.tfidf = {}
        self.words = words
        self.sample_doc_ids = sample_doc_ids
        self.num_sample_docs = len(self.sample_doc_ids)
        self.appearances_file = '..//data//appearances.csv'

    def count_appearances(self):
        """
            Count the number of docs in which each word appears in
            Call once to write to a csv file for later use
            If the list of sample docs changes, rerun this function
        """
        appearances = {}

        for doc_id in self.sample_doc_ids:
            words = brown.words(fileids=[doc_id])
            words = util.preprocess_words(words)

            freq_dict = {}
            util.update_freq_dict(freq_dict, words)

            for word in freq_dict:
                if word in appearances:
                    appearances[word] += 1
                else:
                    appearances[word] = 1

        with open(self.appearances_file, 'w+') as f:
            writer = csv.writer(f, delimiter=' ')
            for word in appearances:
                row = []
                row.append(word)
                row.append(appearances[word])
                writer.writerow(row)

    def normalize_tf(self):
        """
            Normalize the value of term-frequency of each word
            in case tf-idf biases towards long docs
        """
        max_frequency = -1

        for word in self.tf:
            if self.tf[word] > max_frequency:
                max_frequency = self.tf[word]

        for word in self.tf:
            self.tf[word] = 0.5 + 0.5 * self.tf[word] / max_frequency

    def load_idf(self):
        """
            Load the number of appearances from file,
            calculate the idf values of each word and assign
            to the idf dictionary
        """
        if not os.path.isfile(self.appearances_file):
            self.count_appearances()

        with open(self.appearances_file, 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            for row in reader:
                word = row[0]
                appearances = int(row[1])
                self.idf[word] = math.log(
                    self.num_sample_docs / (1 + appearances)
                )

    def calc_tfidf(self):
        self.tf = {}
        util.update_freq_dict(self.tf, self.words)
        self.normalize_tf()
        self.load_idf()

        for word in self.tf:
            if word in self.idf:
                idf_val = self.idf[word]
            else:
                idf_val = math.log(self.num_sample_docs)
            self.tfidf[word] = self.tf[word] * idf_val

    def rate_sentences(self, sentence_texts):
        """
            Rate sentences according to tf-idf value of words
            The score of each sentence is the average tf-idf value of
            each word
        """
        sentences = [Sentence(sentence_texts[i])
                     for i in range(len(sentence_texts))]

        for sentence in sentences:
            for word in sentence.words:
                sentence.score += self.tfidf[word]
            sentence.score /= len(sentence.words)

        sentences.sort(key=lambda x: x.score, reverse=True)

        return sentences

    def best_sentences(self, sentence_texts, k):
        """
            Return k best sentences according to score
        """
        sentences = self.rate_sentences(sentence_texts)

        if k > len(sentences):
            k = len(sentences)

        result = []
        for i in range(k):
            result.append(sentences[i])

        return result
