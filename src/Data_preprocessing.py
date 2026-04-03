from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler , RobustScaler , LabelEncoder,OneHotEncoder
from sklearn.metrics import r2_score , mean_absolute_error , mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from collections import OrderedDict

def Data_preprocessing(df):
    df=df.drop_duplicates()
    
    X = df.drop(columns=['PowerConsumption_Zone2'])
    y = df['PowerConsumption_Zone2']
    numerical_col = X.select_dtypes(exclude='object').columns
    categorical_col = X.select_dtypes(include='object').columns

    X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                        test_size = 0.3,
                                                        random_state=3)


    num_pipeline= Pipeline(steps=[
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",RobustScaler())

    ])

    cat_col = Pipeline(steps=[
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("encoder",OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num",num_pipeline,numerical_col),
        ("cat",cat_col,categorical_col)

    ])

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train , X_test,y_train,y_test