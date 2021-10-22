import argparse
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall, AUC
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from preprocessing import preprocessing

def get_args():
    """
    Read program arguments
    """
    parser = argparse.ArgumentParser(
        description='Train and save model')

    parser.add_argument(
        '--save',
        action='store_true',
        help='Whether to evaluate the test set, otherwise it will only run training and show validation metrics'
    )
    return parser.parse_args()


def get_tokenizer(path):
    """
    Load tokenizer from file
    """
    tokenizer = None
    with open(path, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer


def get_model(max_words, maxlen, emb_dim=8, output=4):
    """
    Returns multiclass classification model
    """
    model = models.Sequential([
        layers.Embedding(max_words, emb_dim, input_length=maxlen),
        layers.Flatten(),
        layers.Dense(output, activation='softmax')  # 4 probability output
    ])
    metrics = [
        CategoricalAccuracy(),
        Precision(),
        Recall(),
        AUC()
    ]
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=metrics)
    return model


def show_evaluation_result(step, result):
    """
    Show result of model.evaluate()
    """
    print('\n\n================== {} ==================='.format(step))
    print('LOSS\t\t: {:.5f}'.format(result[0]))
    print('ACCURACY\t: {:.5f}'.format(result[1]))
    print('PRECISION\t: {:.5f}'.format(result[2]))
    print('RECALL\t\t: {:.5f}'.format(result[3]))
    print('AUC\t\t: {:.5f}'.format(result[4]))


def show_classification_report(y_true, y_pred):
    """
    Show confusion matrix and classification report
    """
    print('\nCONFUSION MATRIX:')
    print(confusion_matrix(y_true=y_true,
                           y_pred=y_pred))

    print('\nCLASSIFICATION REPORT:')
    print(classification_report(y_true=y_true, y_pred=y_pred))


if __name__ == '__main__':
    args = get_args()

    column = 'pesan'
    max_len = 50
    max_words = 20000
    epochs = 20

    # load and do some cleaning
    dataset = pd.read_csv('./data/sms.csv')
    dataset[column] = dataset[column].apply(preprocessing)

    # remove duplication
    dataset.drop_duplicates(subset=[column], inplace=True)

    # shuffles dataset
    shuffled = dataset.sample(frac=1).reset_index(drop=True)

    # load tokenizer
    tokenizer = get_tokenizer('./model/tokenizer.pkl')

    texts = shuffled[column].values
    labels = to_categorical(shuffled['tipe'].values)

    # tokenize texts
    tokens = tokenizer.texts_to_sequences(texts)

    # pad sequences
    data = pad_sequences(tokens, maxlen=max_len)

    # split train & test data
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=85)

    # get a portion of validation data from training data
    X_train2, X_val, y_train2, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=85)

    # create model
    model = get_model(max_words, max_len,
                      emb_dim=8,
                      output=4)

    # train
    hist = model.fit(X_train2, y_train2,
                     epochs=epochs,
                     batch_size=512,
                     validation_data=(X_val, y_val),
                     verbose=0)

    # evaluate using validation data
    val_result = model.evaluate(X_val, y_val)
    if not args.save:
        show_evaluation_result('VALIDATION', val_result)

    else:
        # Create new model and train using the whole training dataset
        model2 = get_model(max_words, max_len,
                           emb_dim=8,
                           output=4)
        model2.fit(X_train, y_train,
                   epochs=epochs,
                   batch_size=512,
                   verbose=0)

        # evaluate model
        test_result = model2.evaluate(X_test, y_test)

        # show both training and test result so we can compare them
        show_evaluation_result('VALIDATION', val_result)
        show_evaluation_result('TEST', test_result)

        predictions = model2.predict(X_test)
        show_classification_report(y_true=np.argmax(y_test, axis=1),
                                   y_pred=np.argmax(predictions, axis=1))

        # save the model?
        save_model = input("Save model to disk? (y/[N]): ")
        if save_model.lower() == 'y':
            model2.save('./model/keras_model.h5')

            new_model=tf.keras.models.load_model("./model/keras_model.h5")
            new_model.summary()

            # Convert the model.
            converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
            tflite_model = converter.convert()

            # Save the model.
            with open('converted_model.tflite', 'wb') as f:
                f.write(tflite_model)   
                print('Model saved to {}'.format('./model/latest'))
