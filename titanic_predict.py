import joblib 
model = joblib.load("titanic_model.pkl")
print(model.predict([[3,1,147]]))  