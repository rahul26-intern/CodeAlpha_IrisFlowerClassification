#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 1. Load the dataset

df = pd.read_csv('Iris.csv')

print("--- Dataset Overview ---")
print(df.head())
print("\nSpecies distribution:\n", df['Species'].value_counts())

# 2. Features and Labels extraction
# 'Id' is just a sequence number, and 'Species' is our target label
X = df.drop(columns=['Id', 'Species'])
y = df['Species']

# 3. Train-Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Model Training (Using Random Forest)
print("\n--- Training Random Forest Model ---")
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# 5. Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Visualization: Confusion Matrix Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(
    confusion_matrix(y_test, y_pred), 
    annot=True, 
    fmt='d', 
    xticklabels=model.classes_, 
    yticklabels=model.classes_, 
    cmap='Blues'
)
plt.title('Iris Classification - Confusion Matrix')
plt.ylabel('Actual Species')
plt.xlabel('Predicted Species')
plt.tight_layout()

# Display the plot
plt.show()


# In[ ]:




