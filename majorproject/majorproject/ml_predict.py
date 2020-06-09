import pickle
import keras
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model

def prob(loaded_model,loaded_tokenizer,q1,q2):
    ques1 = []
    ques1.append(q1)
    ques2 = []
    ques2.append(q2)
    question1_word_sequences = loaded_tokenizer.texts_to_sequences(ques1)
    question2_word_sequences = loaded_tokenizer.texts_to_sequences(ques2)
    q1_data = pad_sequences(question1_word_sequences, maxlen=25)
    q2_data = pad_sequences(question2_word_sequences, maxlen=25)
    X = np.stack((q1_data,q2_data),axis=1)
    y = loaded_model.predict([X[:,0],X[:,1]])

    print(y)
    return y[0][0]*100



def prediction_model(ques1,ques2):
    loaded_model = keras.models.load_model('quora_pairs.h5')

    with open('tokenizer.pickle', 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)
    return prob(loaded_model,loaded_tokenizer,ques1,ques2)