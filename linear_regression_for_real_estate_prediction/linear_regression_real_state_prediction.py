import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as plt1

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Read the csv file
data = pd.read_csv('/real_estate_prediction_dataset/Real estate.csv')

keys = data.keys()

print(keys)

# Dropping the unused column
#data = data.drop(['No'],axis=1)
data = data.loc[:, ~data.columns.isin(['No'])]

# Describe the data
print(data.describe(include='all'))

# Get the output distribution
sns.distplot(data['Y house price of unit area'])
plt.show()
# Get the input features
input_features = data.loc[:, ~data.columns.isin(['Y house price of unit area'])]

# Get the target value
target = data['Y house price of unit area']



# Visualize the input w.r.t output
fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1)

data.plot(y='Y house price of unit area', x='X1 transaction date', kind="scatter", ax=ax1, legend=False)
data.plot(y='Y house price of unit area', x='X2 house age', kind="scatter", ax=ax2, legend=False)
data.plot(y='Y house price of unit area', x='X3 distance to the nearest MRT station', kind="scatter", ax=ax3, legend=False)
data.plot(y='Y house price of unit area', x='X4 number of convenience stores', kind="scatter", ax=ax4, legend=False)
data.plot(y='Y house price of unit area', x='X5 latitude', ax=ax5, kind="scatter", legend=False)
data.plot(y='Y house price of unit area', x='X6 longitude', ax=ax6, kind="scatter", legend=False)

plt.show()




# Visualize the data distribution
plt.figure(1)
input_features.hist(bins=50)

target.hist(bins=50)

plt.show()


# Split the data into train and test

X_train, X_test, y_train, y_test = train_test_split(input_features, target, test_size=0.20, random_state=42)

# Apply the tranformation to data, standardize the data to mean=0, variance=1
# scaler= StandardScaler().fit(X_train)
#
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.fit_transform(X_test)

scaler= StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


'''
print(X_train.mean(axis=0))
print(X_train.std(axis=0))

print(X_test.mean(axis=0))
print(X_test.std(axis=0))
'''

# Fit the Linear Regression model
model = LinearRegression().fit(X_train, y_train)

# get the score
training_score = model.score(X_train, y_train)
testing_score = model.score(X_test, y_test)

# get the prediction and the rmse output
predicted_output_train = model.predict(X_train)
rmse_train = np.sqrt(mean_squared_error(predicted_output_train, y_train))

predicted_output_test = model.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(predicted_output_test, y_test))

print("Model performance with all input features: ")
print("Training : ", training_score, rmse_train)
print("Testing : ", testing_score, rmse_test)


print("Analysis after eliminating each feature one by one")
for key in data.keys():
    print('########')

    print("Model performance after eliminating feature: ", key)
    input_features = data.loc[:, ~data.columns.isin([key, 'Y house price of unit area'])]
    #print(input_features.keys())
    target = data['Y house price of unit area']
    X_train, X_test, y_train, y_test = train_test_split(input_features, target, test_size=0.20, random_state=42)
    # Apply the tranformation to data, standardize the data to mean=0, variance=1
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    # Fit the Linear Regression model
    model = LinearRegression().fit(X_train, y_train)

    # get the score
    training_score = model.score(X_train, y_train)
    testing_score = model.score(X_test, y_test)

    # get the prediction and the rmse output
    predicted_output_train = model.predict(X_train)
    rmse_train = np.sqrt(mean_squared_error(predicted_output_train, y_train))

    predicted_output_test = model.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(predicted_output_test, y_test))

    print("Training : ", training_score, rmse_train)
    print("Testing : ", testing_score, rmse_test)



plt.scatter(y_test, predicted_output_test)
plt.show()
