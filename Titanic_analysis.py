import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the Dataset
titanic_data = pd.read_csv("Data/train.csv")
if titanic_data.empty:
    print("Error: The dataset is empty or not found.")
else:
    print("✓ Dataset loaded successfully.")

print("\n" + "="*60)
print("DATASET OVERVIEW")
print("="*60)

# Display the first few rows of the dataset
print("\n📊 First few rows of the dataset:")
print(titanic_data.head())
print("\n" + "-"*60)
# Display the column names and data types
print("\n📋 Column names and data types:")
print(titanic_data.info())
print("-"*60)

# display basic statistics about the dataset
print("\n📈 Basic statistics about the dataset:")
print(titanic_data.describe())
print("-"*60)

print("\n" + "="*60)
print("DATA CLEANING")
print("="*60)

#check for missing values
print("\n🔍 Missing values in each column:")
print(titanic_data.isnull().sum())
print("-"*60)

# Fill missing Age using the median Age of passengers with the same Sex and Pclass.
# avoid chained-assignment warning by assigning the result back
median_age = titanic_data.groupby(['Sex', 'Pclass'])['Age'].transform('median')
titanic_data['Age'] = titanic_data['Age'].fillna(median_age)
if titanic_data['Age'].isnull().sum() > 0:
    print("❌ Error: Age column still has missing values.")
else:
    print("✓ Age column: Missing values filled successfully.")

# Fill missing Embarked with the most common port of embarkation
mode_embarked = titanic_data['Embarked'].mode()[0]
titanic_data['Embarked'] = titanic_data['Embarked'].fillna(mode_embarked)
if titanic_data['Embarked'].isnull().sum() > 0:
    print("❌ Error: Embarked column still has missing values.")
else:
    print("✓ Embarked column: Missing values filled successfully.")

# Drop the Cabin column due to a large number of missing values
titanic_data.drop('Cabin', axis=1, inplace=True)
if 'Cabin' in titanic_data.columns:
    print("❌ Error: Cabin column was not dropped successfully.")
else:
    print("✓ Cabin column: Dropped successfully.")
print("-"*60)


print("\n" + "="*60)
print("FEATURE ANALYSIS")
print("="*60)

#check for categorical cloumns
print("\n📝 Categorical columns:")
categorical_columns = titanic_data.select_dtypes(include=['object']).columns
print(categorical_columns.tolist())

# Check fro numerical columns
print("\n🔢 Numerical columns:")
numerical_columns = titanic_data.select_dtypes(include=['int64', 'float64']).columns
print(numerical_columns.tolist())


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


print("\n" + "="*60)
print("FEATURE ENCODING")
print("="*60)

# Label encoding of Sex column
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
titanic_data['sex_encoded'] = label_encoder.fit_transform(titanic_data['Sex'])
if 'sex_encoded' not in titanic_data.columns:
    print('\n❌ Error: Sex column not encoded.')
else:
    print('\n✓ Sex column: Label encoding completed.')

# One-hot encoding of embarked column
titanic_data = pd.get_dummies(titanic_data, columns=['Embarked'], drop_first=True)
if 'Embarked_Q' not in titanic_data.columns or 'Embarked_S' not in titanic_data.columns:
    print('❌ Error: Embarked column not one-hot encoded.')
else:
    print('✓ Embarked column: One-hot encoding completed.')
print("-"*60)




# Define feature and Targets
Features = ['Pclass', 'Age', 'sex_encoded', 'Embarked_Q', 'Embarked_S', 'SibSp', 'Parch', 'Fare']
Target = 'Survived'


# Split data for training and testing
x = titanic_data[Features]
y = titanic_data[Target]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state = 42)


# Train the logistic regrassion model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)


# Using trained model to predict the test set
y_prediction = model.predict(x_test)


# Comparing those predictions with the actual survival values
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_prediction))
print("\n📋 Classification Report:")
print(classification_report(y_test, y_prediction))

# Simple Performance Summary
print("\n" + "="*60)
print("FINAL PERFORMANCE SUMMARY")
print("="*60)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_prediction)
print(f"\n🎯 Overall Accuracy: {accuracy:.2%}")
print("   (Percentage of predictions that were correct)")

# Count correct and incorrect predictions
correct = (y_test == y_prediction).sum()
incorrect = (y_test != y_prediction).sum()
print(f"\n✓ Correct Predictions: {correct}")
print(f"❌ Incorrect Predictions: {incorrect}")
print(f"📈 Total Test Cases: {len(y_test)}")
print("\n" + "="*60)
