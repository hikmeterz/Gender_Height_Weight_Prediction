import pandas as pd

import matplotlib.pyplot as plt
  
from sklearn.model_selection import train_test_split

import numpy as np

# Linear Regression class
class LinearRegression:
    def _init_(self):
        self.m1 = 1.0
        self.m2 = 2.0
        self.b = 0.0
        self.loss = []
    
    def fit(self, x, y, learning_rate=0.0000000005, num_iterations=1000): 
        for iteration in range(num_iterations):
            m_gradient = 0.0
            b_gradient = 0.0
            total_error = 0.0
            
            for i in range(len(x)):
                error = y[i] - (self.m1 * x[i][0] + self.m2 * x[i][1] + self.b)
                m_gradient += -2 * x[i][0] * error
                b_gradient += -2 * error
                total_error += error ** 2
            
            self.m1 -= learning_rate * m_gradient
            self.m2 -= learning_rate * m_gradient
            self.b -= learning_rate * b_gradient
            
            mse = total_error / len(x)
            self.loss.append(mse)
            print(f"Iteration {iteration+1}: MSE = {mse}")
    
    def predict(self, x):
        return [self.m1 * xi[0] + self.m2 * xi[1] + self.b for xi in x]


#df = pd.read_csv("500_Person_Gender_Height_Weight_Index.csv")
'''
X = df.iloc[:,1:3].values
  
Y = df.iloc[:,3].values
      
# Splitting dataset into train and test set
  
X_train, X_test, Y_train, Y_test = train_test_split( 
      X, Y, test_size = 1/2, random_state = 0 )
      
# Model training
      

model = LinearRegression()
model.fit(X_train, Y_train)
      
# Pediction on test set
  
Y_pred = model.predict( X_test )

print("Predicted y:", Y_pred)

# Plot the loss graph
plt.plot(range(len(model.loss)), model.loss)
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Loss Graph")
plt.show()


# predicted_index_train = np.round(model.predict(X_train)).astype(int)
predicted_index_test = [int(round(model.predict([xi])[0])) for xi in X_test]

# Calculate train and test accuracies
# train_accuracy = sum(predicted_index_train == Y_train) / len(Y_train)
test_accuracy = sum(predicted_index_test == Y_test) / len(Y_test)
# print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

# # Plot the accuracy graph
# plt.plot(range(len(predicted_index_train)), Y_train, label="Actual Index (Train)")
# plt.plot(range(len(predicted_index_train)), predicted_index_train, label="Predicted Index (Train)")
plt.scatter(range(len(predicted_index_test)), Y_test, label="Actual Index (Test)")
plt.scatter(range(len(predicted_index_test)), predicted_index_test, label="Predicted Index (Test)")
plt.xlabel("Data Points")
plt.ylabel("Index")
plt.title("Actual vs Predicted Index")
plt.legend()
plt.show()
'''