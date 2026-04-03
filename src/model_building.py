
import numpy as np
import pickle
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

def model_building(X_train,X_test,y_train,y_test,pickle_path="best_model.pkl"):
    models=({
         "XGBRegressor":XGBRegressor(n_estimators=100 , 
                                    max_depth=6)
    })
    for model_name , model in models.items():
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        report = r2_score(y_test,y_pred) 

        print(f"model name:{model_name} r2score is :{report:.4f}")

    with open(pickle_path,"wb") as f:
       pickle.dump(model,f)   
    return report , model

