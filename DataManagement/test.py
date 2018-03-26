import os
import sys
import pandas_ml as pdml

module_path = os.path.abspath(os.path.join('..'))
module_path2 = os.path.abspath(os.path.join('../../AxolotlDataScience'))

if module_path not in sys.path:
    sys.path.append(module_path)

if module_path2 not in sys.path:
    sys.path.append(module_path2)

from DataManipulation import DataManipulation as axolotlManipulation
from DataVisualization import TypeOfAccessBarChart
from Graphics import MultipleColumnsBarChartByClass
from Graphics import SingleColumnBarChartByClass
from MachineLearning.Classifiers import ClassifierFactory
from MachineLearning.Classifiers import ClassifierTypes
from Validation import NFoldCrossValidation as validator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from DataPrediction import MLHandler
from DataPrediction import CartolaProML

print(customers_df.tipo.value_counts())

df1 = customers_df[customers_df["tipo"] == 1].sample(400)
df12 = customers_df[customers_df["tipo"] == 1].sample(400)
df13 = customers_df[customers_df["tipo"] == 1].sample(400)
df2 = customers_df[customers_df["tipo"] == 0].sample(400*4)

df_new = pd.concat([df1, df12, df13, df1, df2])

print(df_new.tipo.value_counts())

classifier_types = ClassifierTypes.ClassifierTypes
attributes = df_new.loc[:, df_new.columns != 'tipo']
classes = df_new["tipo"]

total_custumers_len = len(customers_df[customers_df["tipo"] == 1])
total_not_custumers_len = len(customers_df[customers_df["tipo"] == 0])


ml_algorithms = [classifier_types.KNN, classifier_types.Tree, classifier_types.NaiveBayes]

ml_handler = MLHandler.MLHandler(attributes, classes)
trained_ml_algorithms = ml_handler.train_data_in_mls_algorithms(ml_algorithms)
accuracy_ml_algorithms = ml_handler.get_accuracy_of_mls_algorithms(trained_ml_algorithms)
prediction_ml_algorithms = ml_handler.get_prediction_classes_of_mls_algorithms(trained_ml_algorithms, attributes=customers_df.loc[:, customers_df.columns != 'tipo'],
                                                      classes=customers_df["tipo"])


false_positives_and_negatives = ml_handler.get_false_negatives_and_positives(prediction_ml_algorithms,
                                  data_frame['user'], attributes=customers_df.loc[:, customers_df.columns != 'tipo'],
                                                      classes=customers_df["tipo"])

print(total_custumers_len, len(false_positives_and_negatives["FN"]))


print(accuracy_ml_algorithms)

print("FP: ", len(false_positives_and_negatives["FP"]), "/", total_not_custumers_len, " = ",
      len(false_positives_and_negatives["FP"]) / total_not_custumers_len,
      "\nFN: ", len(false_positives_and_negatives["FN"]), "/", total_custumers_len, " = ",
      len(false_positives_and_negatives["FN"]) / total_custumers_len)
