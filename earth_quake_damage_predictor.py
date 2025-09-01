#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing libraries
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from category_encoders import OneHotEncoder
from category_encoders import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split


# In[ ]:


#Loading sql
get_ipython().run_line_magic('load_ext', 'sql')
get_ipython().run_line_magic('sql', 'sqlite:///../nepal.sqlite')


# In[ ]:


# wranvgle function for wrangling and importing data from sql
def wrangle(dbpath):
    conn = sqlite3.connect(dbpath)
    query = """
    SELECT DISTINCT(id_map.building_id) AS b_id, building_structure.*, building_damage.damage_grade
    FROM id_map
    JOIN building_structure ON id_map.building_id = building_structure.building_id
    JOIN building_damage ON building_structure.building_id = building_damage.building_id
    WHERE district_id=3
    """
    df = pd.read_sql(query, conn, index_col="b_id")
    #encode damage_grade column to binary
    df["damage_grade"] = df["damage_grade"].str[-1].astype(int)
    df["severe_damage"] =  (df["damage_grade"] > 3).astype(int)
    # Add leaky columns to columns to be dropped
    dropcols=[col for col in df.columns if "post" in col]
    #append damage_grade to dropcols
    dropcols.append("damage_grade")
    #append count_floors_pre_eq to dropcols due to its contribution to multicollinearity
    dropcols.append("count_floors_pre_eq")
    #add "building_id" to dropcols since it has high cardinality and does not really serve any purpose
    dropcols.append("building_id")
    df = df.drop(columns=dropcols)
    return df


# In[ ]:


#loading data using the wrangle function
df = wrangle("/app/nepal.sqlite")
df.head()


# In[ ]:


#Plot correlation heatmap to investigate multi collinearity
correlation = df.drop(columns="severe_damage").select_dtypes("number").corr()
sns.heatmap(correlation, annot=True, cmap= "hot")


# In[ ]:


#investigating high cardinality features
df.nunique()
df.info()


# In[ ]:


fig, ax = plt.subplots() 

# Calculate value counts and plot on the axes object
df["severe_damage"].value_counts(normalize=True).plot(
    kind="bar",
    ax=ax  # Direct the plot to our Axes object
)

# Set labels and title 
ax.set_xlabel("Severe Damage")
ax.set_ylabel("Relative Frequency")
ax.set_title("Kavrepalanchok, Class Balance");


# In[ ]:


#B0x plot of the severe_daamge distribution
fig, ax = plt.subplots() 

# Create the Seaborn boxplot 
sns.boxplot(x=df["severe_damage"], y=df["plinth_area_sq_ft"], ax=ax)

# Set labels and title
ax.set_xlabel("Severe Damage")
ax.set_ylabel("Plinth Area [sq. ft.]")
ax.set_title("Kavrepalanchok, Plinth Area vs Building Damage");


# In[ ]:


roof_pivot = pd.pivot_table(
    df, index = "roof_type", values = "severe_damage", aggfunc=np.mean
).sort_values(by="severe_damage")
roof_pivot


# In[3]:


#Vertical split into X and y
X = df.drop(columns="severe_damage")
y = df["severe_damage"]
print("X shape:", X.shape)
print("y shape:", y.shape)


# In[ ]:


#splitting into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)


# In[ ]:


#getting the baseline accuracy of model
acc_baseline = df["severe_damage"].value_counts(normalize=True).max()
print("Baseline Accuracy:", round(acc_baseline, 2))


# In[ ]:


#Model initiation and fitting
model_lr = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    LogisticRegression(max_iter=1000)
)
model_lr.fit(X_train, y_train)


# In[ ]:


#training and validation accuracy
lr_train_acc = model_lr.score(X_train, y_train)
lr_val_acc = model_lr.score(X_val, y_val)

print("Logistic Regression, Training Accuracy Score:", lr_train_acc)
print("Logistic Regression, Validation Accuracy Score:", lr_val_acc)


# In[ ]:


#Tuning Hyperparameters
depth_hyperparams = range(1, 16)
training_acc = []
validation_acc = []
for d in depth_hyperparams:
    model_dt = make_pipeline(
        OrdinalEncoder(),
        DecisionTreeClassifier(max_depth=d, random_state=42)

    )
    model_dt.fit(X_train, y_train)
    #train score
    training_acc.append(model_dt.score(X_train, y_train))
    #val score
    validation_acc.append(model_dt.score(X_val, y_val))


# In[ ]:


#printing the series of validation accuracies
submission = pd.Series(validation_acc, index=depth_hyperparams).sort_values(ascending=False)


# In[ ]:


#plotting the training and validation accuracies for each depth
fig, ax = plt.subplots() 

#  Plot the training accuracy on the axes object
ax.plot(list(depth_hyperparams), training_acc,  label="training")

#  Plot the validation accuracy on the same axes object
ax.plot(list(depth_hyperparams),  validation_acc, label="validation") 

#  Set labels and title  
ax.set_xlabel("Max Depth")
ax.set_ylabel("Accuracy Score")
ax.set_title("Validation Curve, Decision Tree Model")

# Add the legend 
ax.legend()


# In[ ]:


#Final Model
final_model_dt = make_pipeline(
    OrdinalEncoder(),
    DecisionTreeClassifier(random_state=42, max_depth=10)
)
final_model_dt.fit(X, y)


# In[ ]:


#Testing the model
X_test = pd.read_csv("data/kavrepalanchok-test-features.csv", index_col="b_id")
y_test_pred = final_model_dt.predict(X_test)
y_test_pred[:5]


# In[ ]:


feature_names=X.columns
importances= final_model_dt.named_steps["decisiontreeclassifier"].feature_importances_


# In[ ]:


#Plottting the feature importance
fig, ax = plt.subplots() 

# Create the horizontal bar plot on the axes object
feat_imp.plot(kind="barh", ax=ax)

# Set labels and title 
ax.set_xlabel("Gini Importance")
ax.set_ylabel("Feature")
ax.set_title("Kavrepalanchok Decision Tree, Feature Importance")

# Apply tight layout 
fig.tight_layout()


# In[ ]:





# In[ ]:




