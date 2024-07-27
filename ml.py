#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Load the csv file into a DataFrame
df = pd.read_csv(r"C:\Users\HP\Desktop\NS.csv")

# Display the first few rows of the DataFrame
encoder = LabelEncoder()
df['User'] = encoder.fit_transform(df['User'])
df['Letter'] = encoder.fit_transform(df['Letter'])
df['Group'] = encoder.fit_transform(df['Group'])
df = df[['User',
         'Letter',
         'Group',
         'g1 Hold Latency','g1 Intergroup Latency','g1 Press Latency','g1 Release Latency',
         'g2 Hold Latency','g2 Intergroup Latency','g2 Press Latency','g2 Release Latency',
         'g3 Hold Latency','g3 Intergroup Latency','g3 Press Latency','g3 Release Latency',
         'g4 Hold Latency','g4 Intergroup Latency','g4 Press Latency','g4 Release Latency'
        ]]

X = df.iloc[:,1:]
y = df.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Define a pipeline
numeric_features = X.columns
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)])

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', LogisticRegression(multi_class='multinomial', max_iter=1000))])

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Calculate correlation
correlation_matrix = df.corr()

# Plot heatmap for correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# In[ ]:




