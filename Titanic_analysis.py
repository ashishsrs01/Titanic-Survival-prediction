import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the Dataset
titanic_data = pd.read_csv("Data/train.csv")

# Display the first few rows of the dataset
print(titanic_data.head())
# Display the column names and data types
print(titanic_data.info())
# display basic statistics about the dataset
print(titanic_data.describe())

#check for missing values
print("\nCount of missing values in each column:")
print(titanic_data.isnull().sum())

# Fill missing Age using the median Age of passengers with the same Sex and Pclass.
# avoid chained-assignment warning by assigning the result back
median_age = titanic_data.groupby(['Sex', 'Pclass'])['Age'].transform('median')
titanic_data['Age'] = titanic_data['Age'].fillna(median_age)

# Fill missing Embarked with the most common port of embarkation
mode_embarked = titanic_data['Embarked'].mode()[0]
titanic_data['Embarked'] = titanic_data['Embarked'].fillna(mode_embarked)

# Drop the Cabin column due to a large number of missing values
titanic_data.drop('Cabin', axis=1, inplace=True)

#check for categorical cloumns
print("\nCategorical columns in the dataset:")
categorical_columns = titanic_data.select_dtypes(include=['object']).columns
print(categorical_columns)
# Check fro numerical columns
print("\nNumerical columns in the dataset:")
numerical_columns = titanic_data.select_dtypes(include=['int64', 'float64']).columns
print(numerical_columns)

# Visualize the distribution of the target variable 'Survived'
sns.countplot(x='Survived', data=titanic_data)
plt.title('Distribution of Survived')
plt.legend(title='Survived', labels=['No', 'Yes'])   
plt.show()

# Visualize the relationship between 'Pclass' and 'Survived'
sns.countplot(x='Pclass', hue='Survived', data=titanic_data)
plt.title('Survival by Passenger Class') 
plt.legend(title='Survived', labels=['No', 'Yes'])   
plt.show()

# Visualize the relationship between "Age" and "Survival"
sns.histplot(data=titanic_data, x='Age', hue='Survived', multiple='stack')
plt.title('Age Distribution by Survival')
plt.legend(title='Survived', labels=['No', 'Yes'])  
plt.show()

#Visualize the realationship between 'Sex' and 'Survival'
sns.countplot(x='Sex', hue='Survived', data=titanic_data)
plt.title('Survival by Sex')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()