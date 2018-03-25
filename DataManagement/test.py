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
from Graphics import MultipleColumnsBarChartByClass
from Graphics import SingleColumnBarChartByClass
from MachineLearning.Classifiers import ClassifierFactory
from MachineLearning.Classifiers import ClassifierTypes
from Validation import NFoldCrossValidation as validator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_frame = pd.read_csv('../input_data/ge_mes_olimpiadas.csv', sep=',')
cartola_df = data_frame.copy()

axolotl_manipulation = axolotlManipulation.DataManipulation(cartola_df)

to_remove_columns = ["home_notic_olimp", "ginastica", "basquete", "volei", "handebol", "tenis", "atletismo", "natacao",
                     "judo",
                     "saltos_orn", "canoagem", "home", "cartola_pro", "tipo", "user", "dias_desktop", "pviews_desktop",
                     "visitas_desktop", "dias_mobile", "pviews_mobile",
                     "visitas_mobile", 'tempo_total_desktop', 'tempo_total_mobile']

age_range = [-np.inf, 13, 23, 35, 48, np.inf]
dias_range = axolotl_manipulation.get_discretization_intervals_based_on_number_of_groups('dias', 5)
pviews_range = axolotl_manipulation.get_discretization_intervals_based_on_number_of_groups('pviews', 10)
visitas_range = axolotl_manipulation.get_discretization_intervals_based_on_number_of_groups('visitas', 10)
tempo_total_range = axolotl_manipulation.get_discretization_intervals_based_on_number_of_groups('tempo_total', 10)

axolotl_manipulation = axolotl_manipulation.drop_columns(to_remove_columns) \
                                           .set_categorical_columns(['sexo', 'uf']) \
                                           .divide_by_column(['fut_int', 'futebol', 'fut_olimp', 'blog_cartola'], 'tempo_total') \
                                           .normalize_columns(['fut_int_by_tempo_total', 'futebol_by_tempo_total', 'fut_olimp_by_tempo_total', 'blog_cartola_by_tempo_total'])\
                                           .discretize_column('idade', age_range) \
                                           .discretize_column('dias', dias_range) \
                                           .discretize_column('pviews', pviews_range) \
                                           .discretize_column('visitas', visitas_range) \
                                           .discretize_column('tempo_total', tempo_total_range)

fut_int_by_tempo_total_range = axolotl_manipulation.get_discretization_intervals_based_on_number_of_groups('fut_int_by_tempo_total', 10)
futebol_by_tempo_total_range = axolotl_manipulation.get_discretization_intervals_based_on_number_of_groups('futebol_by_tempo_total', 10)
fut_olimp_by_tempo_total_range = axolotl_manipulation.get_discretization_intervals_based_on_number_of_groups('fut_olimp_by_tempo_total', 10)
blog_cartola_by_tempo_total_range = axolotl_manipulation.get_discretization_intervals_based_on_number_of_groups('blog_cartola_by_tempo_total', 10)


axolotl_manipulation = axolotl_manipulation.discretize_column('fut_int_by_tempo_total', fut_int_by_tempo_total_range) \
                                           .discretize_column('futebol_by_tempo_total', futebol_by_tempo_total_range) \
                                           .discretize_column('fut_olimp_by_tempo_total', fut_olimp_by_tempo_total_range) \
                                           .discretize_column('blog_cartola_by_tempo_total', blog_cartola_by_tempo_total_range)

cartola_df = axolotl_manipulation.get_data_frame()

cartola_df['cartola'].fillna(0)
cartola_df['idade'].fillna(0, inplace=True)

classifier_types = ClassifierTypes.ClassifierTypes
factory = ClassifierFactory.ClassifierFactory()

attributes = cartola_df.loc[:, cartola_df.columns != 'cartola'].columns.values.tolist()
print(cartola_df.isnull().sum())

learning_algorithm = factory.choose_classifier(classifier_types.GRASPForest)

learning_algorithm.set_number_of_trees(1)\
                  .set_number_of_GRASP_randomish(1)\
                  .init_learning_algorithm() \
                  .get_learning_algorithm()

# learning_algorithm.fit(cartola_df.loc[:, cartola_df.columns != 'cartola'], cartola_df["cartola"])

n_fold = validator.NFoldCrossValidation(learning_algorithm, cartola_df.loc[:, cartola_df.columns != 'cartola'], cartola_df["cartola"])
accuracy = n_fold.n_fold_cross_validation()

print(accuracy)
