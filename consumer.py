
from fruitclassifier import FruitClassifier

if __name__ == '__main__':
    fruit_classifier = FruitClassifier()
    fruit_classifier.__init__()
    fruit_classifier.load_model()

    print fruit_classifier.predict()[0]

