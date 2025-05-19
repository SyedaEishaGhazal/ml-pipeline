from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def load_and_preprocess_data():
    housing = fetch_california_housing(as_frame=True)
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target)

    # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
