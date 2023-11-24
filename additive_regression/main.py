import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet


class ProphetForecast:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load_data()
        self.full_data = self.prepare_full_data()
        self.model = self.train_prophet_model()
        self.forecast = self.make_forecast()

    def load_data(self):
        data = pd.read_csv(self.file_path)
        data["Date"] = pd.to_datetime(data["Date"])
        return data

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
        return self.model.predict(future)

    def plot_forecast(self):
        self.model.plot(self.forecast)

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
            if row["yhat"] > 0.05:
                plt.plot(row["ds"], 1, marker="o", markersize=2, color="purple")
                plt.plot([row["ds"], row["ds"]], [0.05, 1], color="purple",
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


def main():
    file_path = "additive_regression/cargo_thefts_dates.csv"
    forecast = ProphetForecast(file_path)
    forecast.plot_forecast()


if __name__ == "__main__":
    main()
