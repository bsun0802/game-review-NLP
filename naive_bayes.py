import pandas as pd
import numpy as np
import string
import re
from collections import defaultdict

from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

class NaiveBayesNLP:
    """Naive Bayes classfier for binary sentiment analysis
        Lidstone smoothing(alpha)
    """

    def __init__(self):
        self.vocab = set()
        self.params = {"wc": {}, "num_tokens": {}, "class_prob": {}}

    def clean(self, text):                                                                                              ### clean and word_frequency need to be refined when feature engineer
        remove = str.maketrans("", "", string.punctuation + string.digits)
        return text.translate(remove).lower()

    def word_frequency(self, clean_text):
        """Laplace +1 smooth"""
        # freq = defaultdict(lambda: 1)
        freq = defaultdict(int)
        for word in re.split("\W+", clean_text):
            self.vocab.add(word)
            freq[word] += 1
        return freq

    def train(self, X, y):
        # class probability
        self.params["class_prob"]["pos"] = np.log(np.mean(y == 1))
        self.params["class_prob"]["neg"] = np.log(np.mean(y == -1))

        # Count(W | Class) and gather vocabulary
        self.params["wc"]["pos"] = self.word_frequency(self.clean(" ".join(X[y == 1])))
        self.params["wc"]["neg"] = self.word_frequency(self.clean(" ".join(X[y == -1])))

        self.params["num_tokens"]["pos"] = sum(self.params["wc"]["pos"].values())
        self.params["num_tokens"]["neg"] = sum(self.params["wc"]["neg"].values())

    def _prob(self, word, direction, alpha=.5):
        return (self.params["wc"][direction][word] + alpha) / (self.params["num_tokens"][direction] + alpha * len(self.vocab) + alpha)

    def predict(self, X, alpha=.5):
        result = []
        for x in X:
            pos, neg = 0, 0
            freqs = self.word_frequency(self.clean(x))
            for word, _ in freqs.items():
                # if word not in self.vocab:
                #     continue
                pos += np.log(self._prob(word, "pos", alpha=alpha))
                neg += np.log(self._prob(word, "neg", alpha=alpha))
            pos += self.params["class_prob"]["pos"]
            pos += self.params["class_prob"]["neg"]
            if pos > neg:
                result.append(1)
            else:
                result.append(-1)
        return result

    def _metrics(self):
        with open("vocab.txt", "w") as o:
            o.write("\n".join(sorted(self.vocab)))

# train = pd.read_csv("train-ns", sep="\t")
# test_file = "dev-ns"
# test = pd.read_csv(test_file, sep="\t")

# train_x = np.array(train["review"])
# train_y = np.array(train["label"])
# test_x = np.array(test["review"])
# test_y = np.array(test["label"])

# nb = NaiveBayesNLP()
# nb.train(train_x, train_y)
# result = nb.predict(test_x)
# print(result) #

# def tokenize(sentence):
#     return [word for word in word_tokenize(sentence) if word.isalpha()]

data = pd.read_csv("review-ascii-only.dev", sep = "\t")
for seed in (1,2,3,4,5):
    X_train, X_test, y_train, y_test = train_test_split(data["review"], data["label"], test_size=.2, random_state=seed)
    print(f" ---------- Cross-validation  :  {seed}  ---------------- ")
    for alpha in (.001,.01, .05, .1,.3, .5):
        nb = NaiveBayesNLP()
        nb.train(X_train, y_train)
        print(f"Accuracy of alpha = {alpha}  :  {np.mean(nb.predict(X_test,alpha=alpha) == y_test)}")
# nb._metrics()
