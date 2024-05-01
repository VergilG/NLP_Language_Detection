import os
import sys
import re
import time
from feature_extraction import run_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import SGDClassifier


def modeling(path, model):
    # Text data and labels
    text = []
    labels = []  # 0 means modern, 1 means classical.

    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.txt'):
                print()
                print('Splitting text...')
                with open(os.path.join(root, file), 'r') as r:
                    r = r.read()
                    t_read = split_sentences(r)
                    n = 0
                    for t in t_read:
                        n += 1
                        text.append(t)
                        if 'classical' in file:
                            labels.append(1)
                        else:
                            labels.append(0)

    for m in model:
        X = run_model(text, m)

        # Split training set and test set
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.1, random_state=42)

        # Training model
        time_std = time.time()
        sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=10, tol=None)
        sgd.fit(X_train, y_train)
        time_end = time.time()
        time_spend = round(time_end - time_std, 2)
        print(f'Training completed, {m}the training time of the model is: {time_spend}s.')

        # Model evaluation
        y_pred = sgd.predict(X_test)

        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))


path_text = '/path/to/processed/files'

if __name__ == '__main__':
    modeling(path_text, ['ngram', 'tfidf', 'bow'])
