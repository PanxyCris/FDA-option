import pandas as pd
from data_process.read_data import read_announcement_dates, fetch_data
from ml_models.deep_learning import DeepLearningMethod
from ml_models.random_forest import RandomForestMethod
from simulate_option.simulation import simulate_option_result
from utils.plot_util import *


def load_raw_data():
    ticker_dates = read_announcement_dates()
    return fetch_data(ticker_dates, features)


def load_data(features, return_column):
    all_data = pd.read_csv('pharm_v1.csv')  # fetch_data(ticker_dates, features) -> load from raw
    all_data = all_data[features + [return_column]].dropna()
    return all_data


def process_model(model, features, return_column):
    all_data = load_data(features, return_column)
    X = all_data[features]
    thresholds = np.arange(0.01, 0.1, 0.001)
    best_threshold, best_combined_metric, best_report, precision_0, precision_1, recall_0, recall_1 = model.grid_search_thresholds(
        X, all_data, thresholds,
        return_column)
    plot_best_threshold_results(model, thresholds, precision_0, precision_1, recall_0, recall_1)
    all_data['Target'] = (all_data[return_column] >= best_threshold).astype(int)
    y = all_data['Target']
    predictions, y_test = model.train_and_predict(X, y, best_threshold)

    print(f"Best Threshold: {best_threshold}")
    print(f"Best Combined Metric: {best_combined_metric}")
    print("Best Classification Report:\n", best_report)
    plot_confusion_matrix(model, y_test, predictions, classes=['Not Reach Threshold', 'Reach Threshold'])


def predict(model1, model2, features, return_column, threshold=0.044):
    all_data = load_data(return_column)
    all_data = all_data[features + [return_column]].dropna()
    X = all_data[features]
    all_data['Target'] = (all_data[return_column] >= threshold).astype(int)
    y = all_data['Target']
    predictions1, y_test1 = model1.train_and_predict(X, y, threshold)
    predictions2, y_test2 = model2.train_and_predict(X, y, threshold)
    plot_roc_curve(model1, model2, y_test1, predictions1, predictions2, threshold)
    model1.save_model()
    model2.save_model()


def label_prediction(model1, model2, features, return_column):
    all_data = load_data(return_column)
    filtered_data = all_data[features + [return_column]].dropna()
    X = filtered_data[features]
    for model in [model1, model2]:
        trained_model = model.load_model(X)
        model.model = trained_model
        y = model.predict(X)
        all_data[f'predict_{model.name}'] = pd.Series(y)
    all_data.to_csv('files/datasource/pharm_v1_predicted.csv')


if __name__ == "__main__":
    # Defined machine learning features
    features = ['volatility_atr', 'volume_adi', 'momentum_rsi', 'trend_macd', 'volume_obv', 'momentum_roc',
                'momentum_wr']
    # Defined Whether Matching Returns Threshold or Not(If higher or lower than a return value)
    return_column = 'Vol_Returns_1'
    # Find Best Threshold
    model1 = DeepLearningMethod()
    model2 = RandomForestMethod()
    process_model(model1, features, return_column)

    # Use best threshold to predict the result
    predict(model1, model2, features, return_column)

    # label machine learning learned results
    label_prediction(model1, model2, features, return_column)

    # simulate option trading
    simulate_option_result()

    # plot risk evaluation
    returns_data = pd.read_csv('files/datasource/returns.csv')
    plot_var_subplots(returns_data)
