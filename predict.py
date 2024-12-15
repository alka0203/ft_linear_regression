import numpy as np
from train import train_model

# theta_0 = 0
# theta_1 = 0

# theta = np.array(2)

inputValue = input("Enter a mileage: ")

if inputValue.isdigit() == False:
    print("Please enter a number")
    exit()
    
theta = train_model()

print(theta)

# need to specify data type from input to add
estimatedPrice = theta[0] + (theta[1] * int(inputValue))

print(estimatedPrice)