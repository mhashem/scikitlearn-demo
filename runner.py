
# print result
# print '- result computed in ' + str(float((end - start))) + ' ms'

from fruitclassifier import FruitClassifier

if __name__ == '__main__':

    classifier = FruitClassifier()

    classifier.__init__()
    classifier.train_classifier()
    classifier.save_model()
