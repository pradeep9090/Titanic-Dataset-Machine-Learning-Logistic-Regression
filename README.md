# Titanic Dataset Machine Learning

This project demonstrates how to use machine learning to predict whether a passenger survived the Titanic disaster based on certain features. The goal is to create a predictive model using Logistic Regression.

## Steps

### 1. **Import Libraries**
   ```python
   import pandas as pd
   import numpy as np
   ```
   - `pandas` is used for data manipulation and analysis.
   - `numpy` is used for numerical operations and array handling.

### 2. **Load the Dataset**
   ```python
   df = pd.read_csv('C:/Users/Pradeep/Documents/CSV_FILES/titanic.csv')
   df
   ```
   - Loads the Titanic dataset from the specified file path into a DataFrame `df`.

### 3. **Check for Missing Values**
   ```python
   df.isnull().sum()
   ```
   - Identifies columns with missing values by counting the `NaN` entries.

### 4. **Handle Missing Data**
   ```python
   df['Age'].fillna(df['Age'].mean(), inplace=True)
   ```
   - Fills the missing values in the `Age` column with the mean value of the column, modifying the DataFrame in place.

### 5. **Encode Categorical Data**
   ```python
   df.replace({'Sex':{'male':0, 'female':1}}, inplace=True)
   ```
   - Encodes the `Sex` column, converting 'male' to 0 and 'female' to 1 for machine learning compatibility.

### 6. **Prepare Features and Target**
   ```python
   X = df[['Pclass', 'Sex', 'Age', 'Fare']]
   y = df[['Survived']]
   ```
   - Selects the feature columns (`Pclass`, `Sex`, `Age`, `Fare`) as `X` and the target variable (`Survived`) as `y`.

### 7. **Split the Data into Training and Testing Sets**
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
   ```
   - Splits the dataset into training and testing sets. 80% of the data is used for training, and 20% is used for testing.

### 8. **Train the Model**
   ```python
   from sklearn.linear_model import LogisticRegression
   lr = LogisticRegression()
   lr.fit(X_train, y_train)
   ```
   - Initializes a logistic regression model and trains it on the training data (`X_train`, `y_train`).

### 9. **Make Predictions**
   ```python
   prediction = lr.predict([[1, 0, 20, 100]])
   prediction
   ```
   - Uses the trained logistic regression model to make a prediction on a new sample with values `[Pclass=1, Sex=0, Age=20, Fare=100]`.

## Conclusion

This project demonstrates the basic steps of preparing data, training a logistic regression model, and making predictions. You can further evaluate the model's performance by using accuracy metrics such as precision, recall, and F1-score.

