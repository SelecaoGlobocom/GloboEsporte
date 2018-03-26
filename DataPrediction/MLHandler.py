from MachineLearning.Classifiers import ClassifierFactory
from MachineLearning.Classifiers import ClassifierTypes
from Validation import NFoldCrossValidation as validator
import operator


class MLHandler:

    def __init__(self, attributes, classes):
        self.attributes = attributes
        self.classes = classes
        self.trained_ml_algorithms = None
        self.ml_algorithms_accuracy = None
        self.ml_algorithms_predictions = None
        self.false_positives_dict = None

    def train_data_in_mls_algorithms(self, ml_algorithms):

        factory = ClassifierFactory.ClassifierFactory()

        trained_ml_algorithms = []

        for ml_algorithm in ml_algorithms:

            learning_algorithm = factory.choose_classifier(ml_algorithm) \
                .init_learning_algorithm()

            learning_algorithm.fit(self.attributes, self.classes)
            trained_ml_algorithms.append(learning_algorithm)

        self.trained_ml_algorithms = trained_ml_algorithms

        return self

    def calc_accuracy_of_mls_algorithms(self, attributes=None, classes=None):

        if attributes is not None:
            self.attributes = attributes
        if classes is not None:
            self.classes = classes

        ml_algorithms_accuracy = []

        for trained_ml_algorithm in self.trained_ml_algorithms:
            n_fold = validator.NFoldCrossValidation(trained_ml_algorithm, self.attributes, self.classes)
            accuracy = n_fold.ten_fold_cross_validation()
            ml_algorithms_accuracy.append(accuracy)

        self.ml_algorithms_accuracy = ml_algorithms_accuracy

        return self

    def calc_prediction_classes_of_mls_algorithms(self, attributes=None, classes=None):

        ml_algorithms_predictions = []

        if attributes is not None:
            self.attributes = attributes
        if classes is not None:
            self.classes = classes

        for trained_ml_algorithm in self.trained_ml_algorithms:
            predict_classes = trained_ml_algorithm.predict(self.attributes)
            ml_algorithms_predictions.append(predict_classes)

        self.ml_algorithms_predictions = ml_algorithms_predictions

        return self

    def calc_false_negatives_and_positives(self, users, attributes=None, classes=None):

        false_negatives = []
        false_positives = []

        if attributes is not None:
            self.attributes = attributes
        if classes is not None:
            self.classes = classes

        for i in range(0, len(self.classes)):

            ml_prediction_class = -1
            is_majority = True

            classes = {0: 0, 1: 0}

            for ml_prediction in self.ml_algorithms_predictions:
                classes[ml_prediction[i]] += 1

            if(classes[0] > len(self.ml_algorithms_predictions)/2) or (classes[1] > len(self.ml_algorithms_predictions)/2):
                is_majority = True

            if classes[0] > classes[1]:
                ml_prediction_class = 0
            else:
                ml_prediction_class = 1

            if ml_prediction_class == 1 and self.classes[i] == 0 and is_majority:
                false_positives.append(users[i])
            elif ml_prediction_class == 0 and self.classes[i] == 1 and is_majority:
                false_negatives.append(users[i])

        keys = ["FP", "FN"]
        values = [false_positives, false_negatives]
        self.false_positives_dict = dict(zip(keys, values))

        return self

    def get_accuracy(self):
        return self.ml_algorithms_accuracy

    def get_false_positives_dict(self):
        return self.false_positives_dict


