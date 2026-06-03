# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

```
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
![Screenshot 2024-10-08 111454](https://github.com/user-attachments/assets/c047b6dc-5320-4e7f-8eb9-95cc83c64628)

```
data.isnull().sum()
```
![Screenshot 2024-10-08 111506](https://github.com/user-attachments/assets/05bc1b8a-c3fc-4cd3-9b5e-cbafea639e0f)

```
missing=data[data.isnull().any(axis=1)]
missing
```
![Screenshot 2024-10-08 111533](https://github.com/user-attachments/assets/96fd3209-a896-45d5-bb81-762c8f48164d)

```
data2=data.dropna(axis=0)
data2
```
![Screenshot 2024-10-08 111553](https://github.com/user-attachments/assets/427aa71f-f871-4a4e-ba0b-42080e25c2d9)

```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![Screenshot 2024-10-08 111616](https://github.com/user-attachments/assets/d062f8c9-ed00-4c81-babe-1650bba1f763)

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![Screenshot 2024-10-08 111629](https://github.com/user-attachments/assets/d5062e57-1090-40e0-a0da-879672ba986d)

```
data2
```
![Screenshot 2024-10-08 111647](https://github.com/user-attachments/assets/b4fe6ce9-6521-4f4f-8522-a2021c38022f)

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![Screenshot 2024-10-08 111721](https://github.com/user-attachments/assets/ea8323ff-1d43-437c-82da-528a08dc51bf)

```
columns_list=list(new_data.columns)
print(columns_list)
```
![Screenshot 2024-10-08 111845](https://github.com/user-attachments/assets/3b8a9a8a-9714-4308-a437-a9d561590afb)

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![Screenshot 2024-10-08 111859](https://github.com/user-attachments/assets/0ebff3ff-22fd-4e83-a69b-4c273e4fd2b1)

```
y=new_data['SalStat'].values
print(y)
```
![Screenshot 2024-10-08 111905](https://github.com/user-attachments/assets/21bde01c-82a5-4734-82da-a0ca9a5aaa12)

```
x=new_data[features].values
print(x)
```
![Screenshot 2024-10-08 111911](https://github.com/user-attachments/assets/041d1600-22d2-4ffd-b2a8-7c4c98c0632e)

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
![Screenshot 2024-10-08 111916](https://github.com/user-attachments/assets/bff49f64-d6d7-46ca-95ed-ac89efec733c)

```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![Screenshot 2024-10-08 111921](https://github.com/user-attachments/assets/aaeea18c-e71d-4f7f-b97b-3f88f5811290)

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![Screenshot 2024-10-08 111926](https://github.com/user-attachments/assets/91f2c91f-76e3-465b-90c1-c8b50f59d560)

```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![Screenshot 2024-10-08 111931](https://github.com/user-attachments/assets/68824b96-1b6c-4de1-a099-49cf5122e537)

```
data.shape
```
![Screenshot 2024-10-08 111936](https://github.com/user-attachments/assets/e80bc496-ae22-49c0-8b91-e106e4513c04)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![Screenshot 2024-10-08 113002](https://github.com/user-attachments/assets/fb756cc6-70bc-4394-b603-8a2e4ca1ddba)

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```

![Screenshot 2024-10-08 112000](https://github.com/user-attachments/assets/12470544-0bea-426e-ba43-66af2b46e3a2)

```
tips.time.unique()
```
![Screenshot 2024-10-08 112013](https://github.com/user-attachments/assets/92a88e12-d605-4f81-9207-edf92cd22cbb)

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![Screenshot 2024-10-08 112019](https://github.com/user-attachments/assets/309d6100-d069-4ba0-baff-c73035b16f5d)

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![Screenshot 2024-10-08 112024](https://github.com/user-attachments/assets/4a2e17fa-9f29-4143-ab73-6457ff40b786)

# RESULT:
Thus, Feature selection and Feature scaling has been used on the given dataset.
