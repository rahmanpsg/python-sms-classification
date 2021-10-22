import re
import string
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

## fungsi untuk menghapus url didalam teks
def remove_URL(text):
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub(r"",text)

## fungsi untuk menghapus tanda baca didalam teks
def remove_punct(text):
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)

## fungsi untuk mengubah teks menjadi huruf kecil
def case_folding(text):
    return text.casefold()

## stopwords
def remove_stopwords(text):
    stopwords = StopWordRemoverFactory().create_stop_word_remover()
    return stopwords.remove(text)

## stemming
def stemming(text):
    stemmer = StemmerFactory().create_stemmer()
    return stemmer.stem(text)

def preprocessing(text):
    text = remove_URL(text)
    text = remove_punct(text)
    text = case_folding(text)
    text = remove_stopwords(text)
    # text = stemming(text)
    return text