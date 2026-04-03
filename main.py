from src.Data_ingestion import data_ingestion
from src.Data_preprocessing import Data_preprocessing
from src.model_building import model_building
def main():
    df = data_ingestion()
    X_train , X_test,y_train,y_test=Data_preprocessing(df)
    report = model_building(X_train,X_test,y_train,y_test,pickle_path="model_best.pkl")
     
main()


