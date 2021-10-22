import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from preprocessing import preprocessing

if __name__ == '__main__':
    input_file = "./data/sms.csv"
    output_file = "model/tokenizer.pkl"
    column = "pesan"
    max_words = 20000

    # read CSV input
    df = pd.read_csv(input_file)

    # preprocessing
    df[column] = df[column].apply(preprocessing)

    # dedup
    df.drop_duplicates(subset=[column], inplace=True)

    # create tokenizer
    tokenizer = Tokenizer(num_words=int(max_words))
    tokenizer.fit_on_texts(df[column].values)

    with open(output_file, 'wb') as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
