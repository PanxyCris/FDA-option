import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split

from MLMethod import MLMethod


class BinaryClassificationModel(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassificationModel, self).__init__()
        self.layer_1 = nn.Linear(input_dim, 64)
        self.layer_2 = nn.Linear(64, 32)
        self.layer_out = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.relu(self.layer_2(x))
        x = self.sigmoid(self.layer_out(x))
        return x


class DeepLearningMethod(MLMethod):

    def __init__(self):
        self.name = "Deep Learning"
    def train_model(self, X_train, y_train, epochs=100, batch_size=64):
        input_dim = X_train.shape[1]
        model = BinaryClassificationModel(input_dim)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values

        # Convert data to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        train_data = TensorDataset(X_train, y_train)
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                loss.backward()
                optimizer.step()
        return model

    def train_and_predict(self, X, y, threshold):
        input_dim = X.shape[1]
        model = BinaryClassificationModel(input_dim)

        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Convert data to PyTorch tensors
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32)
        X_test = torch.tensor(X_test.values, dtype=torch.float32)
        y_test = torch.tensor(y_test.values, dtype=torch.float32)

        # Create dataloaders
        train_data = TensorDataset(X_train, y_train)
        train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)

        # Training loop
        for epoch in range(100):  # Number of epochs
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                loss.backward()
                optimizer.step()

        # Disable gradient calculations for validation
        with torch.no_grad():
            model.eval()  # Set the model to evaluation mode
            probabilities = model(X_test).squeeze()
        self.model = model
        predictions = (probabilities >= 0.5).float()
        return predictions, y_test

    def predict(self, X):
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        with torch.no_grad():
            self.model.eval()
            probabilities = self.model(X_tensor).squeeze()
        return (probabilities >= 0.5).float()

    # Define the grid search function
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

            # Convert to PyTorch tensors
            X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

            # Train the model
            model = self.train_model(X_train_tensor, y_train_tensor)

            # Predict on test set
            model.eval()
            with torch.no_grad():
                probabilities = model(X_test_tensor).squeeze()
                predictions = (probabilities >= 0.5).float()

            # Calculate precision and recall for both classes
            precision_0 = precision_score(y_test, predictions.numpy(), pos_label=0)
            precision_1 = precision_score(y_test, predictions.numpy(), pos_label=1)
            recall_0 = recall_score(y_test, predictions.numpy(), pos_label=0)
            recall_1 = recall_score(y_test, predictions.numpy(), pos_label=1)

            precision_0_list.append(precision_0)
            precision_1_list.append(precision_1)
            recall_0_list.append(recall_0)
            recall_1_list.append(recall_1)

            combined_metric = (precision_0 * precision_1) + (recall_0 * recall_1)

            # Update the best metrics
            if combined_metric > best_combined_metric:
                best_combined_metric = combined_metric
                best_threshold = threshold
                best_report = classification_report(y_test, predictions.numpy())

        return best_threshold, best_combined_metric, best_report, precision_0_list, precision_1_list, recall_0_list, recall_1_list

    def save_model(self):
        torch.save(self.model.state_dict(), 'deep_learning_model.pth')

    def load_model(self, X):
        input_dim = X.shape[1]
        model = BinaryClassificationModel(input_dim)
        model.load_state_dict(torch.load('deep_learning_model.pth'))
        model.eval()
        return model
