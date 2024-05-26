
# Linear Regression class
class LinearRegression:
   
    def __init__(self):
        
        self.m = 1.0 
        self.b = 0.0
        self.loss = []
    
    def fit(self, x, y, learning_rate=0.0000000005, num_iterations=1000): 
        #print("AAA")
        for iteration in range(num_iterations):
            m_gradient = 0.0
            b_gradient = 0.0
            total_error = 0.0
            
            for i in range(len(x)):
                error = y[i] - (self.m * x[i][0] + self.m * x[i][1] + self.b)
                m_gradient += -2 * x[i][0] * error
                b_gradient += -2 * error
                total_error += error ** 2
            
            self.m -= learning_rate * m_gradient
            self.m -= learning_rate * m_gradient
            self.b -= learning_rate * b_gradient
            
            mse = total_error / len(x)
            self.loss.append(mse)
            print(f"Iteration {iteration+1}: MSE = {mse}")
    
    def predict(self, x):
        
        results = [(self.m * xi[0] + self.m * xi[1] + self.b) for xi in x]
        return results
