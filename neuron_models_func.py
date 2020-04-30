import pandas as pd
import numpy as np
from sklearn.utils import class_weight


def create_sentence_avg_vector(stmt, model, num_features=100):
    words=stmt.split()
    dict={}
    #function to average all words vectors in a given paragraph
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0
    for word in words:
        if word in model.wv.vocab:
            nwords = nwords+1
            featureVec = np.add(featureVec, model[word])
        else:
            #print('failed to find word vector')
            try:
                dict[word] += 1
            except KeyError:
                dict[word] = 1
            

    if nwords>0:
        #print('averageing vectors...')
        featureVec = np.divide(featureVec, nwords)
        #print('stmt vector is of shape of: {}'.format(featureVec.shape))
    #print('count of stopwords found is : {}'.format(count_of_stopwords_found))
    return pd.Series(featureVec)


def convert_score_cat_to_numeric(score):
    score_dict={'TRUE':0,'mostly true':1,'half true':2,'misleading':3,'mostly untrue':4,'untrue':5}
    return score_dict.get(score)