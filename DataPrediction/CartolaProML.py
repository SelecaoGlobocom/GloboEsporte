import os
import sys

module_path = os.path.abspath(os.path.join('..'))
module_path2 = os.path.abspath(os.path.join('../../AxolotlDataScience'))

if module_path not in sys.path:
    sys.path.append(module_path)

if module_path2 not in sys.path:
    sys.path.append(module_path2)

from DataManipulation import DataManipulation as axolotlManipulation
from DataVisualization import TypeOfAccessBarChart
from DataPrediction import MLHandler
from Graphics import MultipleColumnsBarChartByClass
from Graphics import SingleColumnBarChartByClass
from MachineLearning.Classifiers import ClassifierFactory
from MachineLearning.Classifiers import ClassifierTypes
from Validation import NFoldCrossValidation as validator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class CartolaProML:

    def get_predictions(self, data_frame):
        cartola_df = data_frame.copy()
        cartola_df['tipo'].fillna("NÃO ASSINANTE", inplace=True)

        axolotl_manipulation = axolotlManipulation.DataManipulation(cartola_df)

        to_remove_columns = ["home_notic_olimp", "ginastica", "basquete", "volei", "handebol", "tenis", "atletismo",
                             "natacao", "judo",
                             "saltos_orn", "canoagem", "home", "tipo", "user", "dias_desktop", "pviews_desktop",
                             "visitas_desktop", "dias_mobile", "pviews_mobile",
                             "visitas_mobile", 'tempo_total_desktop', 'tempo_total_mobile', "cartola"]

        age_range = [-np.inf, 0.1, 13, 23, 35, 48, np.inf]
        dias_range = axolotl_manipulation.get_discretization_intervals_based_on_number_of_groups('dias', 5)
        pviews_range = axolotl_manipulation.get_discretization_intervals_based_on_number_of_groups('pviews', 10)
        visitas_range = axolotl_manipulation.get_discretization_intervals_based_on_number_of_groups('visitas', 10)
        tempo_total_range = axolotl_manipulation.get_discretization_intervals_based_on_number_of_groups('tempo_total',
                                                                                                        10)

        cartola_df = axolotl_manipulation.drop_columns(to_remove_columns) \
            .set_categorical_columns(['sexo', 'uf']) \
            .divide_by_column(['fut_int', 'futebol', 'fut_olimp', 'blog_cartola'], 'tempo_total') \
            .normalize_columns(['fut_int_by_tempo_total', 'futebol_by_tempo_total', 'fut_olimp_by_tempo_total',
                                'blog_cartola_by_tempo_total']) \
            .discretize_column('idade', age_range) \
            .discretize_column('dias', dias_range) \
            .discretize_column('pviews', pviews_range) \
            .discretize_column('visitas', visitas_range) \
            .discretize_column('tempo_total', tempo_total_range)

        cartola_df = axolotl_manipulation.get_data_frame()
        cartola_df['idade'].fillna(0, inplace=True)

        classifier_types = ClassifierTypes.ClassifierTypes
        attributes = cartola_df.loc[:, cartola_df.columns != 'cartola_pro']
        classes = cartola_df["cartola_pro"]

        total_len = len(cartola_df["cartola_pro"])
        total_len_pro = len(cartola_df[cartola_df["cartola_pro"] == 1])
        total_len_not_pro = len(cartola_df[cartola_df["cartola_pro"] == 0])

        ml_algorithms = [classifier_types.Tree, classifier_types.KNN, classifier_types.NaiveBayes,
                         classifier_types.NeuralNetwork]

        ml_handler = MLHandler.MLHandler(attributes, classes)

        trained_ml_algorithms = ml_handler.train_data_in_mls_algorithms(ml_algorithms)
        accuracy_ml_algorithms = ml_handler.get_accuracy_of_mls_algorithms(trained_ml_algorithms)
        prediction_ml_algorithms = ml_handler.get_prediction_classes_of_mls_algorithms(trained_ml_algorithms)
        false_positives_and_negatives = ml_handler.get_false_negatives_and_positives(prediction_ml_algorithms,
                                                                                     data_frame['user'])

        print(accuracy_ml_algorithms)
        print("FP: ", len(false_positives_and_negatives["FP"]), "/", total_len_not_pro, " = ",
                      len(false_positives_and_negatives["FP"]) / total_len_not_pro,
              "\nFN: ", len(false_positives_and_negatives["FN"]), "/", total_len_pro, " = ",
                        len(false_positives_and_negatives["FN"]) / total_len_pro)

        return false_positives_and_negatives
