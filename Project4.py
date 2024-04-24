#!/usr/bin/env python
# coding: utf-8

# In[144]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objs as go

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import r2_score


from IPython.display import Image
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


shelter_cpi = pd.read_csv('/Users/mustafa/Desktop/project4_team2/consumer_price_index_shelter.csv')[['REF_DATE', 'VALUE']].rename(columns={'VALUE': 'Shelter'})
mort_int_cpi = pd.read_csv('/Users/mustafa/Desktop/project4_team2/consumer_price_index_mort_int.csv')[['REF_DATE', 'VALUE']].rename(columns={'VALUE': 'Mortgage_and_Interest'})
gasoline_cpi = pd.read_csv('/Users/mustafa/Desktop/project4_team2/consumer_price_index_gasoline.csv')[['REF_DATE', 'VALUE']].rename(columns={'VALUE': 'Gasoline'})
food_cpi = pd.read_csv('/Users/mustafa/Desktop/project4_team2/consumer_price_index_food.csv')[['REF_DATE', 'VALUE']].rename(columns={'VALUE': 'Food'})
energy_cpi = pd.read_csv('/Users/mustafa/Desktop/project4_team2/consumer_price_index_energy.csv')[['REF_DATE', 'VALUE']].rename(columns={'VALUE': 'Energy'})
all_items_cpi = pd.read_csv('/Users/mustafa/Desktop/project4_team2/consumer_price_index_all_items.csv')[['REF_DATE', 'VALUE']].rename(columns={'VALUE': 'All_items'})
grains_and_oilseeds = pd.read_csv('/Users/mustafa/Desktop/project4_team2/3210009801_databaseLoadingData (2).csv')


# In[145]:


food_cpi.fillna(food_cpi.select_dtypes(include=['number']).mean(), inplace=True)
shelter_cpi.fillna(shelter_cpi.select_dtypes(include=['number']).mean(), inplace=True)
mort_int_cpi.fillna(mort_int_cpi.select_dtypes(include=['number']).mean(), inplace=True)
gasoline_cpi.fillna(gasoline_cpi.select_dtypes(include=['number']).mean(), inplace=True)
energy_cpi.fillna(energy_cpi.select_dtypes(include=['number']).mean(), inplace=True)
grains_and_oilseeds.fillna(grains_and_oilseeds.select_dtypes(include=['number']).mean(), inplace=True)



# In[146]:


# Merge the DataFrames based on the 'REF_DATE' column
df_food_price = pd.merge(grains_and_oilseeds, all_items_cpi, on='REF_DATE', how='left')
df_food_price = pd.merge(df_food_price, energy_cpi, on='REF_DATE', how='left')
df_food_price = pd.merge(df_food_price, food_cpi, on='REF_DATE', how='left')
df_food_price = pd.merge(df_food_price, gasoline_cpi, on='REF_DATE', how='left')
df_food_price = pd.merge(df_food_price, mort_int_cpi, on='REF_DATE', how='left')
df_food_price = pd.merge(df_food_price, shelter_cpi, on='REF_DATE', how='left')


# In[147]:


df_food_price


# In[148]:


df_food_price.info()
df_food_price.columns


# In[149]:


columns_to_drop = ['DGUID', 'SCALAR_FACTOR', 'UOM', 'UOM_ID', 'SCALAR_ID', 'VECTOR', 'COORDINATE', 'STATUS', 'SYMBOL', 'TERMINATED', 'DECIMALS']
df_food_price = df_food_price.drop(columns=columns_to_drop, axis=1)
df_food_price.head()


# In[150]:


df_food_price.columns


# In[151]:


# Convert 'REF_DATE' column to datetime format
df_food_price['REF_DATE'] = pd.to_datetime(df_food_price['REF_DATE'])

# Extract year and month as separate numerical features
df_food_price['Year'] = df_food_price['REF_DATE'].dt.year
df_food_price['Month'] = df_food_price['REF_DATE'].dt.month

# Drop the original 'REF_DATE' column
df_food_price.drop('REF_DATE', axis=1, inplace=True)


# In[152]:


# Convert all column names to lowercase
df_food_price.columns = df_food_price.columns.str.lower().str.replace(' ', '_')

# Print the DataFrame to verify the changes
df_food_price
df_food_price.head()


# In[153]:


df_food_price.columns


# In[154]:


df_food_price['commodity_groups'].unique()


# # Exploratory Analysis

# In[155]:


import pandas as pd
import plotly.express as px



# In[156]:


df_food_price.columns


# In[157]:


df_food_price['geo'].unique()


# In[158]:


df_food_price['commodity_groups'].unique()


# In[159]:


selected_commodity_groups = ['Grains [A11]', 'Oilseeds [A12]']
filtered_df = df_food_price[df_food_price['commodity_groups'].isin(selected_commodity_groups)]


# In[160]:


filtered_df



# In[161]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[162]:


selected_commodity_groups = ['Grains [A11]']
filtered_df = df_food_price[df_food_price['commodity_groups'].isin(selected_commodity_groups)]

plt.figure(figsize=(8, 6))
sns.barplot(data=filtered_df, x='geo', y='value', hue='commodity_groups', ci=None)
plt.title("Comparison of 'Grains [A11]' by region")
plt.xlabel('Region')
plt.ylabel('Value')
plt.show()


# In[163]:


selected_commodity_groups = ['Oilseeds [A12]']
filtered_df = df_food_price[df_food_price['commodity_groups'].isin(selected_commodity_groups)]

plt.figure(figsize=(8, 6))
sns.barplot(data=filtered_df, x='geo', y='value', hue='commodity_groups', ci=None)
plt.title("Comparison of'Oilseeds [A12]' by region")
plt.xlabel('Region')
plt.ylabel('Value')
plt.show()


# In[164]:


import seaborn as sns
import matplotlib.pyplot as plt


plt.figure(figsize=(10, 6))
sns.lineplot(data=filtered_df, x='year', y='value', hue='geo', marker='o')
plt.title("Annual Trends for 'Oilseeds [A12]' by Region")
plt.xlabel('Year')
plt.ylabel('Value')
plt.show()


# In[165]:


plt.figure(figsize=(8, 6))
sns.histplot(data=filtered_df, x='value', bins=20, kde=True, color='blue')
plt.title("Distribution of Values for 'Oilseeds [A12]'")
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()



# In[166]:


selected_commodity_groups = ['Grains [A11]', 'Oilseeds [A12]']
filtered_df = df_food_price[df_food_price['commodity_groups'].isin(selected_commodity_groups)]

# Plot a bar chart
plt.figure(figsize=(8, 6))
sns.barplot(data=filtered_df, x='geo', y='value', hue='commodity_groups', ci=None)
plt.title("Comparison of 'Grains [A11]', 'Oilseeds [A12]' by Region")
plt.xlabel('Region')
plt.ylabel('Value')
plt.show()


# In[167]:


import seaborn as sns
import matplotlib.pyplot as plt


plt.figure(figsize=(10, 6))
sns.lineplot(data=filtered_df, x='year', y='value', hue='commodity_groups', marker='o', style='commodity_groups')
plt.title("Trends Over Time for 'Grains [A11]' and 'Oilseeds [A12]'")
plt.xlabel('Year')
plt.ylabel('Value')
plt.legend(title='Commodity Group')
plt.show()


# In[168]:


plt.figure(figsize=(10, 6))
sns.histplot(data=filtered_df, x='value', hue='commodity_groups', multiple='stack', kde=True)
plt.title("Value Distribution for 'Grains [A11]' and 'Oilseeds [A12]'")
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend(title='Commodity Group')
plt.show()


# In[169]:


plt.figure(figsize=(10, 6))
sns.boxplot(data=filtered_df, x='geo', y='value', hue='commodity_groups')
plt.title("Value Range and Outliers by Region for 'Grains [A11]' and 'Oilseeds [A12]'")
plt.xlabel('Region')
plt.ylabel('Value')
plt.legend(title='Commodity Group')
plt.show()


# In[170]:


# You need to pivot the data first
heatmap_data = filtered_df.pivot_table(index='year', columns='geo', values='value', aggfunc='mean')

plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt=".0f")
plt.title("Heatmap of Average Values per Year and Region")
plt.xlabel('Region')
plt.ylabel('Year')
plt.show()


# # Regression Module

# In[171]:


from sklearn.preprocessing import OneHotEncoder


# Step 1: Prepare the data
X = df_food_price.drop(['value', 'commodity_groups'], axis=1)  # Features
y = df_food_price['value']  # Target variable

# Step 2: One-hot encode the 'geo' column
encoder = OneHotEncoder(sparse=False)
geo_encoded = encoder.fit_transform(df_food_price[['geo']])
geo_df = pd.DataFrame(geo_encoded, columns=encoder.get_feature_names_out(['geo']))

# Combine encoded 'geo' with other features
X = pd.concat([X.drop('geo', axis=1), geo_df], axis=1)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Analyze feature importance (coefficients)
coefficients = model.coef_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

# Sort the DataFrame by coefficient (absolute value)
feature_importance_df['Coefficient'] = feature_importance_df['Coefficient'].abs()  # Take absolute values
feature_importance_df = feature_importance_df.sort_values(by='Coefficient', ascending=False)

# Display the feature importance DataFrame
print(feature_importance_df)


# In[172]:


# Filter the DataFrame 
df_oil = df_food_price[df_food_price['commodity_groups'] == 'Oilseeds [A12]']

# Filter out non-numeric columns and the target variable
X = df_oil.select_dtypes(include=['number']).drop(columns=['value'])

# Target variable
y = df_oil['value']

# Step 4: Display the slope (coefficients)
print(f"Model's slope (coefficients): {model.coef_}")

# Step 5: Display the y-intercept
print(f"Model's y-intercept: {model.intercept_}")

# Step 6: Display the model's best fit line formula
formula = f"y = {model.intercept_} + {' + '.join([f'{model.coef_[i]}*{X.columns[i]}' for i in range(len(X.columns))])}"
print(f"Model's formula: {formula}")

# Create a model with scikit-learn
model = LinearRegression()

# Fit the data into the model
model.fit(X, y)

# Make predictions
predicted_values = model.predict(X)

# Display the predictions
"Predicted values for 'Oilseeds [A12]':", predicted_values


# In[173]:


# Combine actual and predicted values into a DataFrame
comparison_df = pd.DataFrame({'Actual_Value': y, 'Predicted_Value': predicted_values})

# Display the first few rows of the comparison DataFrame
# print(comparison_df.head())

# Compute Mean Squared Error (MSE)

mse = mean_squared_error(y, predicted_values)
print("Mean Squared Error:", mse)

# Compute R-squared (R2) score
from sklearn.metrics import r2_score
r2 = r2_score(y, predicted_values)
print("R-squared (R2) score:", r2)


# In[174]:


# Plot the actual and predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y, predicted_values, color='blue', label='Actual vs. Predicted')
plt.plot(y, y, color='red', label='Perfect Fit')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.title('Actual vs. Oil seeds [A12]. using all 8 features')
plt.legend()
plt.grid(True)
plt.show()
plt.show()


# In[175]:


import statsmodels.api as sm


X = sm.add_constant(X)  # adding a constant
model = sm.OLS(y, X).fit()
print(model.summary())


# In[176]:


import seaborn as sns
import matplotlib.pyplot as plt


features = df_food_price[['all_items', 'energy', 'food', 'gasoline', 'mortgage_and_interest', 'shelter', 'year', 'month']]

# Calculate the correlation matrix
corr_matrix = features.corr()

# Plot the correlation matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix of Predictors')
plt.show()


# In[177]:


#### import seaborn as sns
import matplotlib.pyplot as plt


# only the columns 
selected_features = df_food_price[['energy', 'food', 'gasoline', 'mortgage_and_interest', 'shelter']]

# correlation matrix for the selected features
corr_matrix = selected_features.corr()

# Plot 
plt.figure(figsize=(6, 6))  # Adjust the size as needed
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix of Selected Predictors')
plt.show()


# In[178]:


# Filter the DataFrame 
df_oil = df_food_price[df_food_price['commodity_groups'] == 'Oilseeds [A12]']

# Select the independent variables based on the provided features
X = df_oil[['energy', 'food', 'gasoline', 'mortgage_and_interest', 'shelter']]

# Target variable
y = df_oil['value']

# Create a model with scikit-learn
model = LinearRegression()

# Fit the data into the model
model.fit(X, y)

# Step 4: Display the slope (coefficients)
print(f"Model's slope (coefficients): {model.coef_}")

# Step 5: Display the y-intercept
print(f"Model's y-intercept: {model.intercept_}")

# Step 6: Display the model's best fit line formula
formula = f"y = {model.intercept_} + {' + '.join([f'{model.coef_[i]}*{X.columns[i]}' for i in range(len(X.columns))])}"
print(f"Model's formula: {formula}")

# Compute Mean Squared Error (MSE)

mse = mean_squared_error(y, predicted_values)
print("Mean Squared Error:", mse)

# Compute R-squared (R2) score
from sklearn.metrics import r2_score
r2 = r2_score(y, predicted_values)
print("R-squared (R2) score:", r2)

# Make predictions
predicted_values = model.predict(X)

# Display the predictions
"Predicted values for oil:", predicted_values


# In[179]:


# Plot the actual and predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y, predicted_values, color='blue', label='Actual vs. Predicted')
plt.plot(y, y, color='red', label='Perfect Fit')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.title('Actual vs. Predicted Values. Using features with higher correlation')
plt.legend()
plt.grid(True)
plt.show()


# In[180]:


import statsmodels.api as sm

# Assuming X are your predictors and y is your response variable and X has been augmented with an intercept
X = sm.add_constant(X)  # adding a constant
model = sm.OLS(y, X).fit()
print(model.summary())


# In[181]:


# Filter the DataFrame 
df_oil = df_food_price[df_food_price['commodity_groups'] == 'Oilseeds [A12]']

# Select the independent variables based on the provided features
X = df_oil[['energy', 'gasoline', 'mortgage_and_interest', 'shelter']]

# Target variable
y = df_oil['value']

# Create a model with scikit-learn
model = LinearRegression()

# Fit the data into the model
model.fit(X, y)

# Step 4: Display the slope (coefficients)
print(f"Model's slope (coefficients): {model.coef_}")

# Step 5: Display the y-intercept
print(f"Model's y-intercept: {model.intercept_}")

# Step 6: Display the model's best fit line formula
formula = f"y = {model.intercept_} + {' + '.join([f'{model.coef_[i]}*{X.columns[i]}' for i in range(len(X.columns))])}"
print(f"Model's formula: {formula}")

# Compute Mean Squared Error (MSE)

mse = mean_squared_error(y, predicted_values)
print("Mean Squared Error:", mse)

# Compute R-squared (R2) score
from sklearn.metrics import r2_score
r2 = r2_score(y, predicted_values)
print("R-squared (R2) score:", r2)

# Make predictions
predicted_values = model.predict(X)

# Display the predictions
"Predicted values for oil:", predicted_values


# In[182]:


# Plot the actual and predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y, predicted_values, color='blue', label='Actual vs. Predicted')
plt.plot(y, y, color='red', label='Perfect Fit')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.title('Actual vs. Predicted Values. Using features with higher correlation')
plt.legend()
plt.grid(True)
plt.show()


# In[183]:


import statsmodels.api as sm

# Assuming X are your predictors and y is your response variable and X has been augmented with an intercept
X = sm.add_constant(X)  # adding a constant
model = sm.OLS(y, X).fit()
print(model.summary())


# In[184]:


# Filter the DataFrame 
df_oil = df_food_price[df_food_price['commodity_groups'] == 'Oilseeds [A12]']

# Select the independent variables based on the provided features
X = df_oil[['energy', 'gasoline', 'mortgage_and_interest', ]]

# Target variable
y = df_oil['value']

# Create a model with scikit-learn
model = LinearRegression()

# Fit the data into the model
model.fit(X, y)

# Step 4: Display the slope (coefficients)
print(f"Model's slope (coefficients): {model.coef_}")

# Step 5: Display the y-intercept
print(f"Model's y-intercept: {model.intercept_}")

# Step 6: Display the model's best fit line formula
formula = f"y = {model.intercept_} + {' + '.join([f'{model.coef_[i]}*{X.columns[i]}' for i in range(len(X.columns))])}"
print(f"Model's formula: {formula}")

# Compute Mean Squared Error (MSE)

mse = mean_squared_error(y, predicted_values)
print("Mean Squared Error:", mse)

# Compute R-squared (R2) score
from sklearn.metrics import r2_score
r2 = r2_score(y, predicted_values)
print("R-squared (R2) score:", r2)

# Make predictions
predicted_values = model.predict(X)

# Display the predictions
"Predicted values for oil:", predicted_values


# In[185]:


# Plot the actual and predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y, predicted_values, color='blue', label='Actual vs. Predicted')
plt.plot(y, y, color='red', label='Perfect Fit')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.title('Actual vs. Predicted Values. Using features with higher correlation')
plt.legend()
plt.grid(True)
plt.show()


# In[186]:


import statsmodels.api as sm

# Assuming X are your predictors and y is your response variable and X has been augmented with an intercept
X = sm.add_constant(X)  # adding a constant
model = sm.OLS(y, X).fit()
print(model.summary())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




