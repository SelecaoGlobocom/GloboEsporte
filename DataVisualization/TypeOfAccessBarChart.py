from Graphics import SingleColumnBarChartByClass

class TypeOfAccessBarChart:

    def __init__(self, plt, data_frame):
        self.plt = plt
        self.df = data_frame

    def plot(self, region_filter, title):

        data_frame = self.apply_region_filter_to_data_frame(region_filter)

        bar_chart = SingleColumnBarChartByClass.SingleColumnBarChartByClass(self.plt, rows=1, columns=4, x_size=20,
                                                                            y_size=5)

        bar_chart.set_attribute_values(data_frame.tipo)\
                 .set_class_values(data_frame.device_preference)\
                 .plot_chart('Tipo x Modo de Acesso', 0, 0)

        bar_chart.set_attribute_values(data_frame.cartola) \
                 .set_class_values(data_frame.device_preference) \
                 .plot_chart('Cartola x Modo de Acesso', 0, 1)

        bar_chart.set_attribute_values(data_frame.cartola_pro) \
                 .set_class_values(data_frame.device_preference) \
                 .plot_chart('Cartola Pro x Modo de Acesso', 0, 2)

        bar_chart.set_attribute_values(data_frame.sexo) \
                 .set_class_values(data_frame.device_preference) \
                 .plot_chart('Sexo x Modo de Acesso', 0, 3)

        self.plt.suptitle(title, fontsize=16, y=1.08)
        self.plt.tight_layout()
        self.plt.show()

    def apply_region_filter_to_data_frame(self, filter):

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
        }.get(filter, self.df)
