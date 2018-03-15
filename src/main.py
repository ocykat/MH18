import util
from tfidf import Tfidf


# visit http://www.nltk.org/book/ch02.html for more info
sample_doc_ids = ('ca16', 'cb02', 'cc17', 'cd12', 'ce36',
                  'cf25', 'cg22', 'ch15', 'cj19', 'ck04',
                  'cl13', 'cm01', 'cn15', 'cp12', 'cr06')


def main():
    words = []
    sentences = []
    with open('..//dataset//computer.txt', 'r') as f:
        for line in f:
            words += util.split_words(line)
            sentences += util.split_sentences(line)

    # Tf-idf evaluation
    tfidf = Tfidf(words, sample_doc_ids)
    tfidf.calc_tfidf()

    # Show tf-idf values
    sorted_tfidf = util.sort_dict_by_value(tfidf.tfidf)
    for i in range(len(sorted_tfidf)):
        print('Word: {0:25}; tfidf = {1}'.format(sorted_tfidf[i][0],
                                                 sorted_tfidf[i][1]))

    # Work out summary
    summary = tfidf.best_sentences(sentences, 100)

    for sentence in summary:
        print(sentence.text)
        print("Score: {0}\n".format(sentence.score))

    print("-----------\nDONE")


if __name__ == "__main__":
    main()
