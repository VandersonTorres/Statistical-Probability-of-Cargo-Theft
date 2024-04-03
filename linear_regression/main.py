import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class CargoTheftForecast:
    def __init__(self, file_path):
        self.data = self.load_data(file_path)
        self.X = self.data["month"].values.reshape(-1, 1)
        self.y = self.data["thefts"].values
        self.model = self.model_training()

    def load_data(self, path):
        data = pd.read_csv(path)
        return data

    def model_training(self):
        model = LinearRegression()
        model.fit(self.X, self.y)
        return model

    def to_predict(self, month):
        month_predict = self.model.predict(month)
        return round(month_predict[0])

    def to_plot(self, month, month_predict):
        colors = {
            "real_data": "royalblue",
            "linear_regression": "orangered",
            "prediction": "limegreen",
            "line_of_prediction": "indigo",
            "bar": "black"
        }

        y_rounded = [round(val) for val in self.y]

        plt.figure(figsize=(10, 6))
        plt.bar(self.X.flatten(), y_rounded, color=colors["bar"], label="Dados de Roubos")
        plt.scatter(self.X, y_rounded, color=colors["real_data"], label="Dados reais", zorder=2)
        plt.plot(self.X, self.model.predict(self.X), color=colors["linear_regression"], linewidth=2,
                 label="Regressão Linear")
        plt.plot(month, month_predict, marker="o", color=colors["prediction"],
                 label=f"Previsão para janeiro: {month_predict}")
        plt.plot(np.concatenate((self.X, month)), np.concatenate((y_rounded, [month_predict])),
                 linestyle="--", color=colors["line_of_prediction"], label="Linha de Previsão")
        plt.xlabel("Série histórica", fontweight="bold")
        plt.ylabel("Quantidade de Roubos", fontweight="bold")
        plt.xticks(
            [
                1,   2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,
                13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
            ],
            [
                "Fev22", "Mar22", "Abr22", "Mai22", "Jun22", "Jul22", "Ago22", "Set22", "Out22", "Nov22", "Dez22", "Jan23",
                "Fev23", "Mar23", "Abr23", "Mai23", "Jun23", "Jul23", "Ago23", "Set23", "Out23", "Nov23", "Dez23", "Jan24"
            ]
        )
        legend = plt.legend()
        legend.get_frame().set_facecolor("lightgray")
        plt.title("Previsão de Roubos para janeiro".upper(), fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.show()

    def run_app(self, month):
        month_predict = self.to_predict(month)
        print(f"Previsão de roubos para o mês de janeiro: {month_predict}")
        self.to_plot(month, month_predict)


if __name__ == "__main__":
    file_path = "linear_regression/statistic_data.csv"
    regression = CargoTheftForecast(file_path)

    january = np.array([[24]])
    regression.run_app(january)
