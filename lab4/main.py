import numpy as np
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score

class BagOfWords:
    def __init__(self):
        self.vocabulary = {}
        self.words_in_order = []

    # data: List[List[str]]
    def build_vocabulary(self, data):
        word_id = 0
        for message in data:
            for word in message:
                if word not in self.vocabulary:
                    self.vocabulary[word] = word_id
                    self.words_in_order.append(word)
                    word_id += 1
        print(f"Vocabulary built. Size: {len(self.vocabulary)}")

    # data: List[List[str]]
    def get_features(self, data):
        num_samples = len(data)
        dictionary_length = len(self.vocabulary)
        features = np.zeros((num_samples, dictionary_length), dtype=np.float64)

        print(f"Generating features for {num_samples} samples...")
        for sample_idx, message in enumerate(data):
            for word in message:
                if word in self.vocabulary:
                    word_id = self.vocabulary[word]
                    features[sample_idx, word_id] += 1
        return features


def normalize_data(train_data, test_data, type=None):
    if type == 'standard':
        scaler = preprocessing.StandardScaler()
        scaler.fit(train_data)
        normal_train_data = scaler.transform(train_data)
        normal_test_data = scaler.transform(test_data)
    elif type == 'l1':
        normal = preprocessing.Normalizer(norm='l1')
        normal_train_data = normal.transform(train_data)
        normal_test_data = normal.transform(test_data)
    elif type == 'l2':
        normal = preprocessing.Normalizer(norm='l2')
        normal_train_data = normal.transform(train_data)
        normal_test_data = normal.transform(test_data)
    else:
        normal_train_data = train_data
        normal_test_data = test_data

    return normal_train_data, normal_test_data


def main():
    train_sentences = np.load('data/training_sentences.npy', allow_pickle=True)
    train_labels = np.load('data/training_labels.npy', allow_pickle=True)

    test_sentences = np.load('data/test_sentences.npy', allow_pickle=True)
    test_labels = np.load('data/test_labels.npy', allow_pickle=True)

    bow = BagOfWords()
    bow.build_vocabulary(train_sentences)

    training_features = bow.get_features(train_sentences)
    test_features = bow.get_features(test_sentences)

    normal_training_features, normal_test_features = normalize_data(training_features, test_features, "l2")

    print("SVM training....")
    model = svm.SVC(C=1, kernel='linear')
    model.fit(normal_training_features, train_labels)
    print("SVM training complete.")

    print("\nPredicting...")
    predictions = model.predict(normal_test_features)
    print("Done predicting")

    accuracy = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions, average='binary')

    print(f"\nAccuracy on test set: {accuracy:.4f}")
    print(f"F1-score on test set: {f1:.4f}")

    coefs = model.coef_[0]
    words = bow.words_in_order
    words_and_coefs = list(zip(words, coefs))
    sorted_words = sorted(words_and_coefs, key=lambda x: x[1])

    bad_words = [word for word, coef in sorted_words[:10]]
    good_words = [word for word, coef in sorted_words[-10:]]

    print(f"\nTop 10 words most indicative of non-spam (ham): {good_words}")
    print(f"Top 10 words most indicative of spam: {bad_words}")

if __name__ == "__main__":
    main()

