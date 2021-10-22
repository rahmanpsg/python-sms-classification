import re
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocessing import preprocessing

TOKENIZER_PATH = 'model/tokenizer.pkl'
MODEL_PATH = 'model/keras_model.h5'

# load tokenizer
tokenizer = None
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

# load model
model = load_model(MODEL_PATH)

def predict_text(text):
    labels = {
        0: 'PENIPUAN (SPAM)',
        1: 'JUDI ONLINE (SPAM)',
        2: 'PINJAMAN ONLINE (SPAM)',
        3: 'LAIN-LAIN (HAM)'
    }

    cleaned_text = preprocessing(text)
    tokenized_text = tokenizer.texts_to_sequences([cleaned_text])
    paded_sequences = pad_sequences(tokenized_text, 50)
    
    pred = model.predict(paded_sequences)
    # print(pred)
    return labels.get(np.argmax(pred)), pred[0, np.argmax(pred)]


print(predict_text("bantu pnutupan cckta dgn cicil ssuai kmmpuanbunga 0disc max smpi 50tdk byr sma"))
