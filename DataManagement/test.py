from DataManagement import DataHandler as dataManipulation
from DataVisualization import TypeOfAccessBarChart
from Graphics import MultipleColumnsBarChartByClass
from Graphics import SingleColumnBarChartByClass
import matplotlib.pyplot as plt
import numpy as np

dt = dataManipulation.DataHandler('../input_data/ge_mes_olimpiadas.csv')
dt.time_to_percentage(['tempo_total_desktop', 'tempo_total_mobile'])\
  .define_desktop_and_cellphone_users()

data_frame = dt.get_data_frame()

df_uf = data_frame

df_uf['regiao'] = df_uf['uf']

sudeste = ['Sao Paulo', 'Rio de Janeiro', 'Minas Gerais', 'Espirito Santo']
sul = ['Parana', 'Santa Catarina', 'Rio Grande do Sul']
nordeste = ['Alagoas', 'Bahia', 'Ceara', 'Maranhao', 'Paraiba', 'Pernambuco', 'Piaui', 'Rio Grande do Norte',
                    'Sergipe']
norte = ['Amapa', 'Roraima', 'Amazonas', 'Rondonia', 'Tocantins', 'Para', 'Acre']
centro_oeste = ['Distrito Federal', 'Goias', 'Mato Grosso', 'Mato Grosso do Sul']

df_uf.loc[df_uf['regiao'].isin(sudeste), 'regiao'] = 'sudeste'
df_uf.loc[df_uf['regiao'].isin(sul), 'regiao'] = 'sul'
df_uf.loc[df_uf['regiao'].isin(nordeste), 'regiao'] = 'nordeste'
df_uf.loc[df_uf['regiao'].isin(norte), 'regiao'] = 'norte'
df_uf.loc[df_uf['regiao'].isin(centro_oeste), 'regiao'] = 'centrooeste'

df_uf = df_uf.loc[df_uf['regiao'].notnull()]


bar_chart = SingleColumnBarChartByClass.SingleColumnBarChartByClass(plt, rows=1, columns=1, x_size=25, y_size=5)

bar_chart.set_attribute_values(df_uf.regiao) \
         .set_class_values(df_uf.device_preference) \
         .plot_chart('Idade x Modo de Acesso', 0, 0)

plt.show()