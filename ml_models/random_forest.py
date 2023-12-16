from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from joblib import dump, load
from MLMethod import MLMethod


class RandomForestMethod(MLMethod):

    def __init__(self):
        self.name = "Random Forest"

    def train_model(self, X_train, y_train, n_estimators=100, random_state=42):
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        clf.fit(X_train, y_train)
        return clf

    def grid_search_thresholds(self, X, all_data, thresholds, return_column):
        best_combined_metric = 0
        best_threshold = 0
        best_report = None
        random_state = 42
        precision_0_list = []
        precision_1_list = []
        recall_0_list = []
        recall_1_list = []

        for threshold in thresholds:
            all_data['Target'] = (all_data[return_column] >= threshold).astype(int)
            y = all_data['Target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
            clf = self.train_model(X_train, y_train)
            predictions = clf.predict(X_test)
            # Calculate precision and recall for both classes
            precision_0 = precision_score(y_test, predictions, pos_label=0)
            precision_1 = precision_score(y_test, predictions, pos_label=1)
            recall_0 = recall_score(y_test, predictions, pos_label=0)
            recall_1 = recall_score(y_test, predictions, pos_label=1)

            precision_0_list.append(precision_0)
            precision_1_list.append(precision_1)
            recall_0_list.append(recall_0)
            recall_1_list.append(recall_1)

            combined_metric = (precision_0 * precision_1) + (recall_0 * recall_1)

            # Update the best metrics
            if combined_metric > best_combined_metric:
                best_combined_metric = combined_metric
                best_threshold = threshold
                best_report = classification_report(y_test, predictions)

        return best_threshold, best_combined_metric, best_report, precision_0_list, precision_1_list, recall_0_list, recall_1_list

    def train_and_predict(self, X, y, threshold):
        random_state = 42
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
        clf = self.train_model(X_train, y_train, random_state=random_state)
        self.model = clf
        predictions = clf.predict(X_test)
        return predictions, y_test

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self):
        dump(self.model, 'random_forest_model.joblib')

    def load_model(self, X):
        return load('random_forest_model.joblib')