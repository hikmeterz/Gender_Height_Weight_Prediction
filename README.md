﻿# Gender_Height_Weight_Prediction
## Project Description
This repository contains a linear regression implementation to predict the weight index of a person based on their gender and height. The project uses a dataset containing gender, height, and weight information to train a linear regression model and evaluate its performance.

## Files
- `500_Person_Gender_Height_Weight_Index.csv`: Dataset containing gender, height, weight, and index information.
- `LR.py`: Initial linear regression implementation.
- `LR1.py`: Improved linear regression implementation.
- `report.ipynb`: Jupyter notebook containing the analysis and results.
- `README.md`: This file.

## Dataset Description
The dataset contains the following attributes:
- `Gender`: The gender of the person (Male/Female).
- `Height`: Height of the person in inches.
- `Weight`: Weight of the person in pounds.
- `Index`: Body mass index category (0-4).

## Implementation Details

### `LR.py`
This script contains the initial implementation of the linear regression model. It includes functions to train the model and predict the index based on the given height and weight data. The training is performed using a basic gradient descent algorithm to minimize the mean squared error.

### `LR1.py`
This script contains an improved version of the linear regression model. The improvements include a more efficient implementation of the gradient descent algorithm and additional features for better performance. 

### `report.ipynb`
This Jupyter Notebook performs data analysis and visualizes the performance of the linear regression model. The main objectives of the notebook are:
1. **Data Loading**: Load the dataset from the CSV file.
2. **Data Exploration**: Explore the dataset using descriptive statistics and visualizations.
3. **Model Training**: Train the linear regression model using the dataset.
4. **Model Evaluation**: Evaluate the model's performance using metrics like Mean Squared Error (MSE) and R-squared score.
5. **Visualization**: Visualize the loss over iterations and the comparison between actual and predicted values.

#### Key Features:
- **Loading the Dataset**: Using `pandas` to load the CSV file into a DataFrame.
- **Exploratory Data Analysis (EDA)**: Descriptive statistics and initial exploration of the dataset.
- **Model Training**: Implementing and training the linear regression model.
- **Model Evaluation**: Calculating the MSE and visualizing the loss and predictions.

