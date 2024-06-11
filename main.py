from autots import AutoTS
import pandas as pd
import matplotlib.pyplot as plt

target = 'new_cases'
location = 'Africa'

df = pd.read_csv("./owid-monkeypox-data.csv")


def get_it(target, location, df):

    df['date'] = pd.to_datetime(df['date'])
    df = df[df['location'] == location]
    df_train = df.iloc[:(-30),:]
    df_test = df.iloc[-31:]
    
    model = AutoTS(
        forecast_length=31,
        frequency='infer',
        prediction_interval=0.9,
        ensemble="simple",
        model_list="superfast",
        transformer_list="superfast",
        drop_most_recent=1,
        max_generations=5,
        num_validations=2,
        validation_method="even"
    )
    model = model.fit(
        df_train,
        date_col='date' ,
        value_col=target,
        id_col='location',
    )
    
    prediction = model.predict()
    forecasts_df = prediction.forecast
    forecasts_up, forecasts_low = prediction.upper_forecast, prediction.lower_forecast
    
    final = df_test[['date', target]]
    final = final.reset_index()
    final['forecast'] = forecasts_df.reset_index().iloc[:,1]
    final['upper'] = forecasts_up.reset_index().iloc[:,1]
    final['lower'] = forecasts_low.reset_index().iloc[:,1]
    return final

def get_rmse(final, target):
    rmse = (final['forecast'] - final[target]).tolist()
    rmse = [x*x for x in rmse]
    rmse = pow(sum(rmse) / len(rmse), 0.5)
    return rmse

def get_mae(final, target):
    errors = (final['forecast'] - final[target]).abs().tolist()
    mae = sum(errors) / len(errors)
    return mae

def get_mape(final, target):
    mape = (abs((final['forecast'] - final[target]) / final[target])).tolist()
    mape = sum(mape) / len(mape) * 100
    return mape

def plot_forecast(df, target):
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df[target], label='Actual')
    plt.plot(df['date'], df['forecast'], label='Forecast')
    plt.fill_between(df['date'], df['lower'], df['upper'], color='k', alpha=.15)
    plt.title(f'{target} Forecast')
    plt.xlabel('Date')
    plt.ylabel(target)
    plt.legend()
    plt.show()

def main():
    nc = get_it("new_cases", location, df).drop(['index'],axis = 1)
    tc = get_it("total_cases", location, df).drop(['index'],axis = 1)
    nd = get_it("new_deaths", location, df).drop(['index'],axis = 1)
    td = get_it("total_deaths", location, df).drop(['index'],axis = 1)


    nc.to_csv("new_cases_forecast.csv", index = False)
    tc.to_csv("total_cases_forecast.csv", index = False)
    nd.to_csv("new_deaths_forecast.csv", index = False)
    td.to_csv("total_deaths_forecast.csv", index = False)
    print(nc)
    print("RMSE for new_cases is: ",get_rmse(nc, 'new_cases'))
    print("MAE for new_cases is: ", get_mae(nc, 'new_cases'))
    print("MAPE for new_cases is: ", get_mape(nc, 'new_cases'))
    plot_forecast(nc, 'new_cases')

    print(tc)
    print("RMSE for total_cases is: ",get_rmse(tc, 'total_cases'))
    print("MAE for total_cases is: ", get_mae(tc, 'total_cases'))
    print("MAPE for total_cases is: ", get_mape(tc, 'total_cases'))
    plot_forecast(tc, 'total_cases')

    print(nd)
    print("RMSE for new_deaths is: ",get_rmse(nd, 'new_deaths'))
    print("MAE for new_deaths is: ", get_mae(nd, 'new_deaths'))
    print("MAPE for new_deaths is: ", get_mape(nd, 'new_deaths'))
    plot_forecast(nd, 'new_deaths')

    print(td)
    print("RMSE for total_deaths is: ",get_rmse(td, 'total_deaths'))
    print("MAE for total_deaths is: ", get_mae(td, 'total_deaths'))
    print("MAPE for total_deaths is: ", get_mape(td, 'total_deaths'))
    plot_forecast(td, 'total_deaths')


if __name__ == "__main__":
    main()
