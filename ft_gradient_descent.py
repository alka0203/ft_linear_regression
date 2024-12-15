import numpy as np
from predict import theta
from train import km, price

def ft_gradient_descent(mileage, price, learning_rate):

    for i in range(1000):
        
        # Update θ₀ and θ₁
        estimatePrice = theta[0] + (theta[1] * int(mileage[i]))

        theta[0] = (learning_rate / m) * np.sum(estimatePrice - price[i])
        theta[1] = (learning_rate / m) * np.sum((estimatePrice - price[i]) * mileage[i])
        
    return theta
