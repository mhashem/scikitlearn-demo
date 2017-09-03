from threading import Thread

from sklearn import tree

import pickle

from profile import profile

# 0 -> bumpy
# 1 -> smooth


class FruitClassifier(object):

    def __init__(self):
        self.features = [[35, 1], [75, 1], [90, 1], [140, 1], [130, 1], [150, 0], [170, 0],
                         [180, 0], [101, 1], [145, 1]]
        self.labels = ["banana", "pineapple", "pineapple", "apple", "apple", "orange", "orange",
                       "orange", "apple", "apple"]
        self.clf = tree.DecisionTreeClassifier()
        self.model_name = 'models/' + str("fruit_model.pkl")
        pass

    @profile
    def train_classifier(self):
        self.clf = self.clf.fit(self.features, self.labels)
        self.save_model()

    def async_training(self):
        t = Thread(target=self.train_classifier)
        t.start()
        pass

    def predict(self):
        return self.clf.predict([[150, 0]])

    def predict(self, f1, f2):
        return self.clf.predict([[f1, f2]])

    def save_model(self):
        pkl_classifier_file = open(self.model_name, 'wb')
        pickle.dump(self.clf, pkl_classifier_file)
        pkl_classifier_file.close()

    def load_model(self):
        pkl_classifier_file = open(self.model_name, 'rb')
        self.clf = pickle.load(pkl_classifier_file)
