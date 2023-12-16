import datetime
import json
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from ta import add_all_ta_features
from utils.plot_util import *
from utils.date_util import DateUtilUSA


def read_announcement_dates():
    """
    read annoucement dates and convert into datetime format
    :return:
    """
    with open('files/datasource/pharm.json', 'r') as jsonfile:
        data = json.load(jsonfile)
        for ticker in data:
            for i, date_str in enumerate(data[ticker]):
                date_str_parts = date_str.split()
                if date_str_parts[0][:2] == '20':
                    year = date_str_parts[0][:4]
                    specific_date_str = date_str_parts[0].split('月')
                    month = specific_date_str[0].split('年')[1]
                    day = specific_date_str[1][:-1]
                else:
                    year = 2023
                    month = date_str_parts[0][:-1]
                    day = date_str_parts[1][:-1]
                data[ticker][i] = datetime.datetime(int(year), int(month), int(day))
    return data


def add_technical_indicators(df):
    """
    Calculate technical indicators for yahoo finance data format
    :param df:
    :return:
    """
    df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume")
    return df


def fetch_data(ticker_dates, features):
    """
    Fetch historical data from Yahoo Finance and save to csv
    :param ticker_dates:
    :param features:
    :return:
    """
    date_util = DateUtilUSA()
    res = pd.DataFrame()
    scaler = MinMaxScaler()
    for ticker in ticker_dates:
        try:
            all_data = yf.download(ticker, start='2010-01-01')
        except:
            continue
        if len(all_data) == 0:
            continue
        cur = pd.DataFrame()
        for pharm_date in ticker_dates[ticker]:
            data = all_data[all_data.index <= pharm_date + datetime.timedelta(days=10)]
            # define maximum returns in future X days
            for i in range(1, 3):
                next_day_high = data['High'].shift(-i)
                data[f'Returns_{i}'] = next_day_high / data['Close'] - 1
            # define absolute maximum returns in future X days
            for i in range(1, 3):
                next_day_high = data['High'].shift(-i)
                next_day_low = data['Low'].shift(-i)
                data[f'Vol_Returns_{i}'] = np.maximum(abs(next_day_high / data['Close'] - 1),
                                                      abs(next_day_low / data['Close'] - 1))
                data[f'NextPrice_{i}'] = np.where(
                    abs(next_day_high / data['Close'] - 1) > abs(next_day_low / data['Close'] - 1),
                    next_day_high,
                    next_day_low)
            data = data.dropna()
            if len(data) == 0:
                continue
            try:
                data = add_technical_indicators(data)
            except:
                continue
            data[features] = scaler.fit_transform(data[features])
            filtered_data = data[data.index < pharm_date]
            if not filtered_data.empty:
                last_row = filtered_data.iloc[-1]
                last_row['DateTime'] = filtered_data.index[-1]
                for i in range(1, 3):
                    last_row[f'TradeDateTime_{i}'] = date_util.get_latter_trading_days(last_row['DateTime'], i)
                cur = pd.concat([cur, last_row.to_frame().T], ignore_index=True)
        cur['Ticker'] = ticker
        res = pd.concat([res, cur], ignore_index=True)
    res.to_csv('files/datasource/pharm_v1.csv')
    return res
