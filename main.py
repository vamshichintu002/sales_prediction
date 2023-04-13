# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load data into a Pandas dataframe
data = pd.read_csv('Advertising.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[['TV', 'Radio', 'Newspaper']], data['Sales'], test_size=0.2, random_state=42)

# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Print the model's R-squared score
print('Model R-squared score:', model.score(X_test, y_test))


# evaluate the model
mse = np.mean((y_pred - y_test)**2)
rmse = np.sqrt(mse)
r2 = model.score(X_test, y_test)

print("Mean Squared Error: ", mse)
print("Root Mean Squared Error: ", rmse)
print("R-squared: ", r2)
next_sales = model.predict(new_data[['TV', 'Radio', 'Newspaper']])
# predicting next sale
new_data = pd.DataFrame({'TV': [200], 'Radio': [100], 'Newspaper': [50]})
print("Predicting Next Sale :",next_sales)