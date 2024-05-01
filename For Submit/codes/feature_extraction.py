from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def ngram(text):
    vectorizer = CountVectorizer(ngram_range=(1, 2))

    X = vectorizer.fit_transform(text)

    return X


def BoW(text):
    vectorizer = CountVectorizer()

    X = vectorizer.fit_transform(text)

    return X


def tfidf(text):
    vectorizer = TfidfVectorizer()

    X = vectorizer.fit_transform(text)

    return X


def run_model(text, model):
    if model == 'ngram':
        return ngram(text)
    elif model == 'bow':
        return BoW(text)
    elif model == 'tfidf':
        return tfidf(text)
    else:
        print('No such model can be run. Please check the name.')

