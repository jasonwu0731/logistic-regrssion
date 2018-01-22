import numpy as np
import re
import collections
import io

def read_data(_file, cleaning):
    revs = []
    max_len = 0
    words_list = []
    with io.open(_file, "r",  encoding="ISO-8859-1") as f:
        next(f)
        for line in f:  
            ID, label, sentence = line.split('\t')        
            label_idx = 1 if label=='pos' else 0 # 1 for pos and 0 for neg
            rev = []
            rev.append(sentence.strip())
            
            if cleaning:
                orig_rev = clean_str(" ".join(rev)) 
            else:
                orig_rev = " ".join(rev).lower()
            
            revs.append({'y':label_idx, 'txt':orig_rev})
            words_list += orig_rev.split()
    return revs, words_list

def clean_str(string):
    """
    TODO: Data cleaning
    """  
    return string

def build_vocab(words_list, max_vocab_size=-1):
    """
    TODO: 
        Build a word dictionary, use max_vocab_size to limit the total number of vocabulary.
        if max_vocab_size==-1, then use all possible words as vocabulary. The dictionary should look like:
        ex:
            word2idx = { 'UNK': 0, 'i': 1, 'love': 2, 'nlp': 3, ... }
        
        top_10_words is a list of the top 10 frequently words appeared
        ex:
            top_10_words = ['a','b','c','d','e','f','g','h','i','j']
    """  
    word2idx = {'UNK': 0} # UNK is for unknown word
    top_10_words = []
    return word2idx, top_10_words

def get_info(revs, words_list):
    """
    TODO: 
        First check what is revs. Then calculate max len among the sentences and the number of the total words
        in the data.
    """  
    nb_sent, max_len, word_count = 0, 0, 0
    return nb_sent, max_len, word_count

def data_preprocess(_file, cleaning, max_vocab_size):
    revs, words_list = read_data(_file, cleaning)
    nb_sent, max_len, word_count = get_info(revs, words_list)
    word2idx, top_10_words = build_vocab(words_list, max_vocab_size) 
    # data analysis
    print("Number of words: ", word_count)
    print("Max sentence length: ", max_len)
    print("Number of sentences: ", nb_sent)
    print("Number of vocabulary: ", len(word2idx))
    print("Top 10 most frequently words", top_10_words)

    return revs, word2idx

def feature_extraction_bow(revs, word2idx):
    """
    TODO: 
        Convert sentences into vectors using BoW. 
        data should be a 2-D array with the size (nb_sentence*nb_vocab)
    """  
    data = []
    label = []
    for sent_info in revs:
        label.append([sent_info['y']])

    return np.array(data), np.array(label)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Building Interactive Intelligent Systems')
    parser.add_argument('-f','--file', help='input csv file', required=False, default='./twitter-sentiment.csv')
    parser.add_argument('-c','--clean', help='True to do data cleaning, default is False', action='store_true')
    parser.add_argument('-mv','--max_vocab', help='max vocab size predifined, no limit if set -1', required=False, default=-1)
    args = vars(parser.parse_args())
    print(args)

    revs, word2idx = data_preprocess(args['file'], args['clean'], int(args['max_vocab']))

    data, label = feature_extraction_bow(revs, word2idx)