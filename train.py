import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from decimal import Decimal
# from ft_gradient_descent import ftt_gradient_descent

theta = np.zeros(2)
stopping_threshold = 1e-100
	# naive prediction

def gradient_descent(mileage, price, learning_rate, x, y):
    per_mse = None
    
    count = len(mileage)

    for i in range(10000):

        # Update θ₀ and θ₁
        estimatePrice = theta[0] + (theta[1] * (x))

        tmpTheta0 = theta[0] - ((learning_rate / count) * np.sum(estimatePrice - y))
        tmpTheta1 = theta[1] - ((learning_rate / count) * np.sum((estimatePrice - y) * x))

        theta[0] = tmpTheta0
        theta[1] = tmpTheta1
        
        cur_mse = (np.sum(estimatePrice - y) ** 2) / count
        
        # denormalizeMSE = mse * price.std() + price.mean()

        # print("MSE: ", denormalizeMSE)
        
        if per_mse is not None and abs(per_mse - cur_mse) <= stopping_threshold:
            print("per_mse: ", per_mse)
            print("cur_mse: ", cur_mse)
            print("theta0: ", theta[0])
            print("theta1: ", theta[1])
            break

        per_mse = cur_mse

    return theta

def train_model():
    data = pd.read_csv("data.csv")

    learning_rate = 0.001

    # reshape to ensure one column with multiple rows
    km = data['km']
    #.values.reshape((-1, 1))
    price = data['price']
    
    standardized_km = (km - km.mean()) / km.std()
    standardized_price = (price - price.mean()) / price.std()
    
    # print("KM: ", standardized_km)
    
    # x = np.array([i for i in km])
    # y = np.array([i for i in price])
    
    x = standardized_km
    y = standardized_price
    
    boop = gradient_descent(km, price, learning_rate, x, y)
    
    denormalized_theta1 = boop[1] * price.std() / km.std()
    
    denormalized_theta0 = boop[0] * price.std() + price.mean() - (denormalized_theta1 * km.mean())
    
    # boop[0] = denormalized_theta0
    # boop[1] = denormalized_theta1

    return denormalized_theta0, denormalized_theta1

# model = LinearRegression().fit(km, price)

# r_sq = model.score(km, price)

# print(f"coefficient of determination: {r_sq}")

# # y_pred = model.predict(km)
# y_pred = model.intercept_ + model.coef_ * km
# print(f"predicted response:\n{y_pred}")
