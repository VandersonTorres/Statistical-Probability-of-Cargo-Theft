from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet
import tkinter as tk
from tkinter import messagebox
from tkinter import simpledialog


class ProphetForecast:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load_data()
        self.full_data = self.prepare_full_data()
        self.model = self.train_prophet_model()
        self.forecast = None

    def load_data(self):
        data = pd.read_csv(self.file_path)
        data["Date"] = pd.to_datetime(data["Date"])
        return data

    def append_data(self, new_date, new_value):
        try:
            new_date = pd.to_datetime(new_date).strftime("%Y-%m-%d")
            new_value = int(new_value)
        except ValueError:
            print("Formato de data ou valor inválido.")
            return

        if pd.to_datetime(new_date) in pd.to_datetime(self.data["Date"]).values:
            overwrite = input(f"Já existe uma entrada para {new_date}. Deseja sobrescrever? (S/N): ")
            if overwrite.lower() != "s":
                print("Operação cancelada.")
                return
            self.data = self.data[self.data["Date"] != new_date]

        new_data = pd.DataFrame({"Date": [new_date], "Value": [new_value]})
        self.data = pd.concat([self.data, new_data], ignore_index=True)
        self.data["Date"] = pd.to_datetime(self.data["Date"])
        self.data["Value"] = pd.to_numeric(self.data["Value"])
        self.data = self.data.sort_values("Date")
        self.full_data = self.prepare_full_data()
        self.model = self.train_prophet_model()
        self.data.to_csv(self.file_path, index=False)
        print("Dados atualizados com sucesso!")

    def prepare_full_data(self):
        date_range = pd.date_range(start=self.data["Date"].min(), end=self.data["Date"].max(), freq="D")
        full_dates = pd.DataFrame({"Date": date_range})
        full_dates["Date"] = pd.to_datetime(full_dates["Date"])
        full_data = pd.merge(full_dates, self.data, on="Date", how="left").fillna(0)
        full_data.columns = ["ds", "y"]
        return full_data

    def train_prophet_model(self):
        model = Prophet()
        model.fit(self.full_data)
        return model

    def make_forecast(self, periods=45):
        future = self.model.make_future_dataframe(periods=periods)
        self.forecast = self.model.predict(future)
        return self.forecast

    def plot_forecast(self):
        if self.forecast is None:
            self.make_forecast()

        plt.figure(figsize=(10, 6))

        self.model.plot(self.forecast, ax=plt.gca())

        thefts_dates = self.data[self.data["Value"] > 0].copy()
        thefts_dates["thefts_amount"] = thefts_dates["Value"]

        future_index = self.forecast[self.forecast["ds"] > self.data["Date"].max()].index.tolist()

        for idx, row in thefts_dates.iterrows():
            plt.plot([row["Date"], row["Date"]], [0, row["thefts_amount"]],
                     color="red", linestyle="--", alpha=0.7, linewidth=0.5)

        plt.fill_between(self.forecast["ds"].iloc[future_index],
                         self.forecast["yhat_lower"].iloc[future_index],
                         self.forecast["yhat_upper"].iloc[future_index],
                         color="lightgreen", alpha=0.3, label="Probability std deviation")

        plt.plot(self.forecast["ds"].iloc[future_index], self.forecast["yhat"].iloc[future_index],
                 color="green", label="Predict", linewidth=1)

        for index, row in self.forecast.iloc[future_index].iterrows():
            if row["yhat"] >= 0.04:
                plt.plot(row["ds"], 1, marker="o", markersize=2, color="purple")
                plt.plot([row["ds"], row["ds"]], [0.04, 1], color="purple",
                         linestyle="-", alpha=0.3, linewidth=0.3)
                plt.text(row["ds"], 0.5, row["ds"].strftime("%Y-%m-%d"),
                         rotation=90, ha="center", color="purple")

        plt.xlabel("PERIOD", fontsize=12, fontweight="bold", color="black", labelpad=10)
        plt.ylabel("VALUES", fontsize=12, fontweight="bold", color="black", labelpad=10)
        plt.yticks([0, 0.05, 0.10, 1])
        plt.ylim(-0.3, 1.1)
        plt.legend(loc="upper left", bbox_to_anchor=(0, 0.89),
                   shadow=True, ncol=1, facecolor="dimgray")

        plt.show()

    def run_forecasting_interface(self):
        while True:
            last_date = datetime.now() - timedelta(days=1)
            autofill_last_date = last_date.strftime("%Y-%m-%d")
            print(
                "1. Visualizar/ Atualizar dados históricos\n"
                "2. Fazer previsão\n"
                "3. Sair"
            )

            choice = input("Escolha uma opção: ")

            if choice == "1":
                print(self.data)
                update = input("Deseja atualizar os dados? (S/N): ")
                if update.lower() == "s":
                    new_date = input("Insira a data da ocorrência: (YYYY-MM-DD) ")
                    new_value = input("Insira a quantidade de ocorrências: ")
                    self.append_data(new_date, new_value)

                    if last_date.strftime("%Y-%m-%d") not in self.data["Date"].astype(str).values:
                        print(
                            "1. Consulta atual\n"
                            "2. Consulta retroativa"
                        )
                        query_identifier = input("Escolha uma opção: ")
                        if query_identifier == "1":
                            self.append_data(autofill_last_date, 0)

                    days = int(input("Insira o número de dias para a previsão: "))
                    self.make_forecast(periods=days)
                    self.plot_forecast()

            elif choice == "2":
                if last_date.strftime("%Y-%m-%d") not in self.data["Date"].astype(str).values:
                    print(
                        "1. Consulta atual\n"
                        "2. Consulta retroativa"
                    )
                    query_identifier = input("Escolha uma opção: ")
                    if query_identifier == "1":
                        self.append_data(autofill_last_date, 0)

                days = int(input("Insira o número de dias para a previsão: "))
                self.make_forecast(periods=days)
                self.plot_forecast()

            elif choice == "3":
                print("SESSÃO FINALIZADA PELO USUÁRIO.")
                break

            else:
                print("Escolha inválida. Tente novamente.")

    @classmethod
    def from_file(cls, file_path):
        return cls(file_path)


class UserInterface:
    def __init__(self, forecast):
        self.forecast = forecast

        self.root = tk.Tk()
        self.root.title("Previsão de Ocorrências")

        self.label = tk.Label(self.root, text="Escolha uma opção:")
        self.label.pack()

        self.button_view_update = tk.Button(
            self.root, text="Visualizar/Atualizar Dados", command=self.view_update_data
        )
        self.button_view_update.pack()

        self.button_make_forecast = tk.Button(self.root, text="Fazer Previsão", command=self.make_forecast)
        self.button_make_forecast.pack()

        self.button_exit = tk.Button(self.root, text="Sair", command=self.exit_program)
        self.button_exit.pack()

    def view_update_data(self):
        data = self.forecast.data
        messagebox.showinfo("Dados Históricos", data.to_string())

        update = messagebox.askyesno("Atualizar Dados", "Deseja atualizar os dados?")
        if update:
            new_date = simpledialog.askstring("Nova Data", "Insira a data da ocorrência (YYYY-MM-DD):")
            new_value = simpledialog.askinteger("Nova Quantidade", "Insira a quantidade de ocorrências:")
            self.forecast.append_data(new_date, new_value)

            last_date = self.forecast.data["Date"].max().strftime("%Y-%m-%d")
            if last_date not in self.forecast.data["Date"].astype(str).values:
                query_identifier = messagebox.askquestion(
                    "Consulta Retroativa", "Deseja fazer uma consulta retroativa?"
                )
                if query_identifier == "yes":
                    self.forecast.append_data(last_date, 0)

            days = simpledialog.askinteger("Previsão", "Insira o número de dias para a previsão:")
            self.forecast.make_forecast(periods=days)
            self.forecast.plot_forecast()

    def make_forecast(self):
        last_date = self.forecast.data["Date"].max().strftime("%Y-%m-%d")
        if last_date not in self.forecast.data["Date"].astype(str).values:
            query_identifier = messagebox.askquestion("Consulta Retroativa", "Deseja fazer uma consulta retroativa?")
            if query_identifier == "yes":
                self.forecast.append_data(last_date, 0)

        days = simpledialog.askinteger("Previsão", "Insira o número de dias para a previsão:")
        self.forecast.make_forecast(periods=days)
        self.forecast.plot_forecast()

    def exit_program(self):
        self.root.destroy()

    def start(self):
        self.root.mainloop()


if __name__ == "__main__":
    file_path = "additive_regression/cargo_thefts_dates.csv"
    forecast = ProphetForecast.from_file(file_path)
    gui = UserInterface(forecast)
    gui.start()


if __name__ == "__main__":
    file_path = "additive_regression/cargo_thefts_dates.csv"
    forecast = ProphetForecast.from_file(file_path)
    forecast.run_forecasting_interface()
