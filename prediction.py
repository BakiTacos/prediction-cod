import joblib

def predict(data):
    xgb = joblib.load("xgboost_model.sav")
    return xgb.predict(data)