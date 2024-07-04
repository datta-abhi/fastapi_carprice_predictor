import joblib
import pandas as pd
import numpy as np
import uvicorn
from fastapi import FastAPI
from car_data_model import Car

# sklearn imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer


# initialising FASTAPI app
app = FastAPI(title = 'Car price prediction',
              description = 'This API will predict Price of car from given set of specifications')

@app.get('/')
def index():
    return 'Welcome to our Car Price Prediction Service'

@app.post('/predict')
def predict(car: Car):
    # passing inputs as attributes of Car basemodel
    x_new = pd.DataFrame(data = dict(
                                    company = [car.company],
                                    year = [car.year],
                                    owner=[car.owner],
                                    fuel=[car.fuel],
                                    km_driven=[car.km_driven],
                                    mileage_mpg=[car.mileage_mpg],
                                    engine_cc=[car.engine_cc],
                                    seats=[car.seats]
    ))
    
    #importing joblib model
    model = joblib.load("carprice_model.joblib")
    predicted_value = model.predict(x_new)[0]
    return 'Car of given specification will cost Rs.{}'.format(np.round(predicted_value,-3))
