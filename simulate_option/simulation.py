import pandas as pd
import QuantLib as ql
from utils.plot_util import *
from simulate_option.option_price import sell_option, buy_option


def read_list_option(type='all'):
    if type == 'weekly' or type == 'monthly':
        file_name = f'option_{type}'
    else:
        file_name = 'option'
    with open(f'files/datasource/{file_name}.txt', 'r') as f:
        data = f.readlines()
        data = [item.replace('\n', '') for item in data]
        return data


def simulate_option_result():
    all_data = pd.read_csv('files/datasource/pharm_v1_predicted.csv')
    weekly_list = read_list_option('weekly')
    monthly_list = read_list_option('monthly')
    all_data['option_frequency'] = 'no_option'
    all_data.loc[all_data['Ticker'].isin(weekly_list), 'option_frequency'] = 'weekly'
    all_data.loc[all_data['Ticker'].isin(monthly_list), 'option_frequency'] = 'monthly'
    all_data = all_data[all_data['option_frequency'] != 'no_option']
    all_data['DateTime'] = pd.to_datetime(all_data['DateTime'])
    all_data = all_data[all_data['DateTime'].dt.year >= 2023]
    all_data.sort_values(by='DateTime', inplace=True)
    balances = [10000 for _ in range(5)]
    # all_data = all_data[(all_data['predict_Deep Learning'] == 1) | (all_data['predict_Random Forest'] == 1)]
    for i in range(5):
        all_data[f'cumulative_returns_scenario{i + 1}'] = 0.0
    all_data.reset_index(drop=True, inplace=True)
    for j, row in all_data.iterrows():
        strike_price = int(row['Close'])  # In The Money
        call_option = buy_option(ql.Option.Call, row['Ticker'], row['Close'], strike_price,
                                 row['DateTime'],
                                 row['option_frequency'])
        put_option = buy_option(ql.Option.Put, row['Ticker'], row['Close'], strike_price,
                                row['DateTime'],
                                row['option_frequency'])
        for i in range(5):
            if (i == 0 and (row['predict_Deep Learning'] or row['predict_Random Forest'])) \
                    or (i == 1 and row['predict_Deep Learning'] and row['predict_Random Forest']) \
                    or (i == 2 and row['predict_Deep Learning']) \
                    or (i == 3 and row['predict_Random Forest']) \
                    or (i == 4):
                hundred_num = 10000 // ((call_option.option_price + put_option.option_price) * 100)
                balances[i] -= hundred_num * 100 * (
                        call_option.option_price + put_option.option_price)  # at least buy 100
                call_profit = sell_option(call_option, row['NextPrice_1'], row['TradeDateTime_1'])
                put_profit = sell_option(put_option, row['NextPrice_1'], row['TradeDateTime_1'])
                balances[i] += hundred_num * 100 * (call_profit + put_profit)
            all_data.at[j, f'cumulative_returns_scenario{i + 1}'] = balances[i] / 10000 - 1
    plot_returns(all_data)
    all_data.to_csv('files/datasource/returns.csv')
    # balance_list.append(balance / 10000 - 1)
    # returns = balance / 10000 - 1
    # print(row['Ticker'], row['DateTime'], balance, returns)
    # print(balance_list)
    # print(np.mean(balance_list))
