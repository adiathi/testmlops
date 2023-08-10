from catboost import CatBoostClassifier
from seldon_core.user_model import SeldonComponent

class CatBoostModel(SeldonComponent):
    def __init__(self, model_path):
        self.model = CatBoostClassifier()
        self.model.load_model(model_path)

    def predict(self, X, features_names):
        # Your preprocessing logic here
        # X is a 2D array of input features

        # Make predictions using the CatBoost model
        predictions = self.model.predict(X)

        return predictions.tolist()

if __name__ == "__main__":
    # Path to your saved CatBoost model
    model_path = "catboost_model.cbm"
    model = CatBoostModel(model_path)
    model.predict_api()
