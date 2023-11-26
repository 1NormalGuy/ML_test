import matplotlib.pyplot as plt
import numpy as np

# Set figure size
plt.figure(figsize=(6, 4))  

# Generate simulated data   
X = np.arange(10)
y = 2 * X + 1   

# Set number of folds to 4
k = 4
fold_size = len(X) / k 

# Plot  
plt.plot(X, y, marker='o', markersize=10)

for i in range(k):
    # Determine index of validation set     
    start = int(i * fold_size)
    end = int((i+1) * fold_size)
    validate = np.arange(start, end)  
    
    # Plot validation set
    plt.plot(X[validate], y[validate], color='grey', marker='x', markersize=15)

plt.title("4-Fold Cross Validation")  
plt.xlabel('X')
plt.ylabel('y')
    
plt.show()