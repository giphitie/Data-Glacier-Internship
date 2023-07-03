import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import pickle

# Load the dataset
data = pd.read_csv('Airplane_data.csv')

# Encode categorical variables
le = LabelEncoder()
data['Group'] = le.fit_transform(data['Group'])
data['Plane'] = le.fit_transform(data['Plane'])
data['Thrower'] = le.fit_transform(data['Thrower'])

# Split the dataset into features and target variable
X = data[['Group', 'Observation', 'Plane', 'Thrower']]
y = data['Distance']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the decision tree regression model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# Saving model to disk
pickle.dump(model, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[2, 4, "Arrow","A1"]]))
