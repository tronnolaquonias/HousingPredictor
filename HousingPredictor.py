import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split, GridSearchCV

# Load the dataset
dataframe = pd.read_csv("Housing.csv")

# Check for missing values in the dataset
# print(dataframe.isnull().sum())

# Convert categorical variables to numerical
dataframe[['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
           'airconditioning', 'prefarea']] = dataframe[['mainroad',
            'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
            ].replace({'yes': 1, 'no': 0})

# Turning all categorical columns to type integer
dataframe[['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
           'airconditioning', 'prefarea']] = dataframe[['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
           'airconditioning', 'prefarea']].astype(int)

# Map the 'furnishingstatus' column to numerical values
furnishingMapping = {'furnished': 0,
                     'semi-furnished': 1,
                     'unfurnished': 2}
dataframe['furnishingstatus'] = dataframe['furnishingstatus'].replace(furnishingMapping).astype(int)

x = dataframe.drop("price", axis=1)
y = dataframe['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=50, test_size=0.25)

# Define the parameter grid for GridSearchCV
different_lengths = [1, 10, 100]
param_grid = [
    {
     'kernel': [C(1.0, (1e-3, 1e4)) * RBF(length, (1e-2, 1e2)) for length in different_lengths],
     'alpha': [1e-2, 1e-3, 1e-4]
    }
]

# Initialize the Gaussian Process Regressor with GridSearchCV
gp = GaussianProcessRegressor(n_restarts_optimizer=50)
grid_search = GridSearchCV(gp, param_grid, cv=None, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get the best model from GridSearchCV
best_gp = grid_search.best_estimator_

# Predict on the test set and get standard deviation
y_pred, sigma = best_gp.predict(X_test, return_std=True)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color ='blue', label='Predicted vs Actual')

# Finding min and max values to scale the graph 
min_value = min(min(y_test), min(y_pred))
max_value = max(max(y_test), max(y_pred))

# Black dashed line for y = x (to easily see perfect predictions)
plt.plot([min_value, max_value], [min_value, max_value], color = "black", linestyle = "--", label="y = x")  

# Format axis labels and limits
plt.ticklabel_format(style='plain')
plt.xlim(min_value, max_value)
plt.ylim(min_value, max_value)

plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Predicted vs Actual Prices')
plt.legend(loc = 'upper right')
plt.show()

