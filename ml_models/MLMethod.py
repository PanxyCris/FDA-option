class MLMethod():
    def train_model(self, X_train, y_train, n_estimators=100, random_state=42):
        raise NotImplementedError("Subclasses should implement this method.")

    def grid_search_thresholds(self, X, all_data, thresholds, return_column):
        raise NotImplementedError("Subclasses should implement this method.")

    def train_and_predict(self, X, y, threshold):
        raise NotImplementedError("Subclasses should implement this method.")

    def save_model(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def load_model(self, X):
        raise NotImplementedError("Subclasses should implement this method.")