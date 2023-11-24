import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

data = pd.read_csv("additive_regression/cargo_thefts_dates.csv")
data["Date"] = pd.to_datetime(data["Date"])
date_range = pd.date_range(start=data["Date"].min(), end=data["Date"].max(), freq="D")

full_dates = pd.DataFrame({"Date": date_range})
full_dates["Date"] = pd.to_datetime(full_dates["Date"])
full_data = pd.merge(full_dates, data, on="Date", how="left").fillna(0)
full_data.columns = ["ds", "y"]

model = Prophet()
model.fit(full_data)
future = model.make_future_dataframe(periods=45)
forecast = model.predict(future)
thefts_dates = data[data["Value"] > 0].copy()
thefts_dates["thefts_amount"] = thefts_dates["Value"]

fig = model.plot(forecast)

for idx, row in thefts_dates.iterrows():
    plt.plot([row["Date"], row["Date"]], [0, row["thefts_amount"]],
             color="red", linestyle="--", alpha=0.7, linewidth=0.5)

future_index = forecast[forecast["ds"] > data["Date"].max()].index.tolist()

plt.fill_between(forecast["ds"].iloc[future_index],
                 forecast["yhat_lower"].iloc[future_index],
                 forecast["yhat_upper"].iloc[future_index],
                 color="lightgreen", alpha=0.3, label="Probability std deviation")

plt.plot(forecast["ds"].iloc[future_index], forecast["yhat"].iloc[future_index],
         color="green", label="Predict", linewidth=1)

for index, row in forecast.iloc[future_index].iterrows():
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
