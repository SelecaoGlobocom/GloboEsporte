import pandas as pd
from DataManipulation import DataManipulation
import numpy as np


class DataHandler:

    def __init__(self, csv_name):
        self.df = pd.read_csv(csv_name, sep=',')
        self.data_manipulation = DataManipulation.DataManipulation(self.df)

    def time_spent_avg_by_page_view(self):

        self.df[['tempo_total']].div(self.df.pviews_desktop, axis=0)
        self.df[['tempo_total']].div(self.df.pviews_mobile, axis=0)

        return self

    def time_spent_avg_by_visit(self):

        self.df[['tempo_total']].div(self.df.visitas_desktop, axis=0)
        self.df[['tempo_total']].div(self.df.visitas_mobile, axis=0)

        return self

    def time_spent_avg_by_day(self):

        self.df[['tempo_total']].div(self.df.dias_desktop, axis=0)
        self.df[['tempo_total']].div(self.df.dias_mobile, axis=0)

        return self

    def discretize_column(self, column_name, interval):
        self.data_manipulation.discretize_column(column_name, interval)

    def time_to_percentage(self, time_variables):

        time_variables_relative = [s + '_relative' for s in time_variables]

        self.df[time_variables_relative] = self.df[time_variables].div(self.df.tempo_total, axis=0)

        return self

    def time_to_interest(self):

        time_variables = ['fut_int', 'ginastica', 'basquete', 'volei', 'handebol', 'tenis',
                          'atletismo', 'futebol', 'blog_cartola', 'fut_olimp', 'judo', 'saltos_orn', 'canoagem']

        for variable in time_variables:
            self.time_to_interest_discretize(variable)

    def time_to_interest_discretize(self, column_name):

        interests = [0.01, 0.5, 1, 2]
        mean_value = self.df[column_name].mean()

        interest_for_this_activity = [x * mean_value for x in interests]

        interest_for_this_activity.insert(0, -np.inf)
        interest_for_this_activity.insert(len(interest_for_this_activity), np.inf)

        self.data_manipulation.discretize_column(column_name, interest_for_this_activity)

    def define_desktop_and_cellphone_users(self):

        time_spent_desktop = self.df['tempo_total_desktop_relative']
        user_preference = []

        for user_time_spent_desktop in time_spent_desktop:

            if user_time_spent_desktop >= 0.7:
                user_preference.append('Desktop')
            elif (user_time_spent_desktop > 0.3) and (user_time_spent_desktop < 0.7):
                user_preference.append('Both')
            else:
                user_preference.append('Mobile')

        self.df['device_preference'] = user_preference

        return self

    def filter_by_region(self, region_name):

        sudeste = ['Sao Paulo', 'Rio de Janeiro', 'Minas Gerais', 'Espirito Santo']
        sul = ['Parana', 'Santa Catarina', 'Rio Grande do Sul']
        nordeste = ['Alagoas', 'Bahia', 'Ceara', 'Maranhao', 'Paraiba', 'Pernambuco', 'Piaui', 'Rio Grande do Norte',
                    'Sergipe']
        norte = ['Amapa', 'Roraima', 'Amazonas', 'Rondonia', 'Tocantins', 'Para', 'Acre']
        centro_oeste = ['Distrito Federal', 'Goias', 'Mato Grosso', 'Mato Grosso do Sul']

        return {
            'sudeste': self.df.loc[self.df['uf'].isin(sudeste)],
            'sul': self.df.loc[self.df['uf'].isin(sul)],
            'nordeste': self.df.loc[self.df['uf'].isin(nordeste)],
            'norte': self.df.loc[self.df['uf'].isin(norte)],
            'centro_oeste': self.df.loc[self.df['uf'].isin(centro_oeste)]
        }.get(region_name, self.df)

    def get_data_frame(self):

        return self.df
