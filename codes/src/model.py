import os
import sys
import re
import time
from feature_extraction import run_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import SGDClassifier


def split_sentences(text):
    # Use regular expressions to match Chinese sentence terminators to segment sentences
    sentence_delimiters = re.compile(u'[。？！；]\s*')
    sentences = sentence_delimiters.split(text)
    # Filter out empty sentences
    sentences = [s for s in sentences if s.strip()]
    return sentences


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
                        progress_percentage = round(n / len(t_read) * 100)
                        print('\rSplit progress: {}%'.format(progress_percentage), '▓' * (progress_percentage // 2),
                              end='')
                        sys.stdout.flush()

    # Text vectorization
    print()
    for m in model:
        print('Text vectorization in progress')
        X = run_model(text, m)
        print('Vectorization is complete.')

        # Split training set and test set
        print('Splitting training and test sets.')
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.1, random_state=42)
        print('Split completed.')

        # Training model
        print('Training model.')
        time_std = time.time()
        sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=10, tol=None)
        sgd.fit(X_train, y_train)
        time_end = time.time()
        time_spend = round(time_end - time_std, 2)
        print(f'Training completed, {m}The training time of the model is: {time_spend}s.')

        # Model evaluation
        print('The trained model is being used for prediction.')
        y_pred = sgd.predict(X_test)
        print('Prediction completed.')
        with open('/Users/Nan/Documents/Lancaster/Term2/SCC413/FinalReport/codes/result/result.txt', 'a') as result:
            result.write(m)
            result.write('\n')
            result.write("Accuracy: ")
            result.write(str(accuracy_score(y_test, y_pred)))
            result.write('\n')
            result.write("Classification Report:")
            result.write('\n')
            result.write(classification_report(y_test, y_pred))
            result.write('\n')

        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))


path_text = '/Users/Nan/Documents/Lancaster/Term2/SCC413/FinalReport/codes/data/processed/final'

if __name__ == '__main__':
    modeling(path_text, ['ngram', 'tfidf', 'bow'])
