from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import joblib
from data_preprocessing import load_and_preprocess_data

X_train, X_test, y_train, y_test = load_and_preprocess_data()

model = SVR()
model.fit(X_train, y_train)

preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)

print(f"MSE: {mse:.2f}")

joblib.dump(model, "svm_model.pkl")
