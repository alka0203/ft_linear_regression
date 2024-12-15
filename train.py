import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from decimal import Decimal
# from ft_gradient_descent import ftt_gradient_descent

theta = np.zeros(2)

def gradient_descent(mileage, price, learning_rate, x, y):
    
    count = mileage.size

    for i in range(count):
        
        # Update θ₀ and θ₁
        estimatePrice = theta[0] + (theta[1] * (x))

        theta[0] -= (learning_rate / count) * np.sum(estimatePrice - y)
        theta[1] -= (learning_rate / count) * np.sum((estimatePrice - y) * x)
        
    return theta


def train_model():
    data = pd.read_csv("data.csv")

    learning_rate = 0.1

    # reshape to ensure one column with multiple rows
    km = data['km'].values.reshape((-1, 1))
    price = data['price']
    
    x = np.array([i for i in km])
    y = np.array([i for i in price])

    return gradient_descent(km, price, learning_rate, x, y)

# model = LinearRegression().fit(km, price)

# r_sq = model.score(km, price)

# print(f"coefficient of determination: {r_sq}")

# # y_pred = model.predict(km)
# y_pred = model.intercept_ + model.coef_ * km
# print(f"predicted response:\n{y_pred}")
