# -Predictive-Modeling-of-Breast-Cancer-Diagnosis-Using-Cell-Characteristics-A-Random-Forest-Approach
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = pd.read_csv("C:/Users/USER/Desktop/Kisii University 2023 students projects/Jefferson Mutinda/data.csv")  # Replace 'your_dataset.csv' with your file path

# Display the first few rows of the dataset
print(data.head())

# Display column names
print(data.columns)

# Bar plot for diagnosis
data['diagnosis'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Diagnosis Distribution')
plt.xlabel('Diagnosis')
plt.ylabel('Frequency')
plt.show()

# Objective 1
# Recode "diagnosis" variable to 0 (benign) and 1 (malignant)
data['diagnosis'] = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)

# Fit logistic regression model
model1 = sm.Logit(data['diagnosis'], sm.add_constant(data[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean']]))
result = model1.fit()
print(result.summary())

# Objective 2
model2 = RandomForestClassifier(n_estimators=500, random_state=123)
model2.fit(data[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean']], data['diagnosis'])

# Objective 3
model3 = RandomForestClassifier(n_estimators=100, random_state=123)
model3.fit(data[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean']], data['diagnosis'])
print(model3.feature_importances_)
