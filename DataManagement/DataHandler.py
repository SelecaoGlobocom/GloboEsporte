import pandas as pd
from DataManipulation import DataManipulation as axolotlManipulation
import numpy as np


class DataHandler:

    def __init__(self, csv_name):
        self.df = pd.read_csv(csv_name, sep=',')
        self.data_manipulation = axolotlManipulation.DataManipulation(self.df)

    def discretize_column(self, column_name, interval):
        self.data_manipulation.discretize_column(column_name, interval)

    def time_to_percentage(self, time_variables):

        time_variables_relative = [s + '_relative' for s in time_variables]

        self.df[time_variables_relative] = self.df[time_variables].div(self.df.tempo_total, axis=0)

        return self

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


    def get_data_frame(self):
        return self.df
