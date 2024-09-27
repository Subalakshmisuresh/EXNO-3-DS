## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Encoding for the feature in the data set.

STEP 4:Apply Feature Transformation for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

```py
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```

![Screenshot 2024-09-27 153640](https://github.com/user-attachments/assets/71e6a7f4-e38e-4d6d-8fb1-39cb69392512)


```py
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

![image](https://github.com/user-attachments/assets/2870fb39-9180-4e4c-9277-b2e976d95edf)


```py
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/d6c1d240-2f97-447a-91c9-9a4a724466ab)


```py
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/5a1ae230-6802-4e72-8fe8-dec5a8cd95f6)

```py
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/6775effe-3f1a-4864-8b5c-0ea1de27b7c4)

```py
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/53b1c974-bedc-4431-8976-c54e59f00625)

```py
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/62c0eb29-7280-43ed-8b34-b7407ff2668c)

```py
pip install --upgrade category_encoders
```
![image](https://github.com/user-attachments/assets/ccf97a11-f3d9-4afe-bd45-327810366662)

```py
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
fb=pd.concat([df,nd],axis=1)
dfb=df.copy()
dfb
```
![image](https://github.com/user-attachments/assets/bdaf3f43-947d-4a31-b1fd-2e5ae23fbdb4)

```py
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![image](https://github.com/user-attachments/assets/7d6bff57-2c19-45d1-bfba-33c40a4fb208)

```py
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/4adf6171-3b1d-49a4-bb7f-2dad03376bc9)

```py
df.skew()
```
![image](https://github.com/user-attachments/assets/b69f9b02-af45-4407-8325-74ad58fdff32)

```py
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/05f2a752-7d03-4434-9b2a-1b30ed04895e)

```py
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/5190c295-f243-46ae-9d71-162b9643b2a3)

```py
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/879769cb-58f2-457a-80c0-e08bd5171d02)

```py
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/5a392a3f-9d7d-4f69-bb32-879306b12d11)

```py
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/e4b74371-b091-43bd-91b8-bf57f216c4b1)

```py
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/7761516e-b6ba-449d-a49b-7b50bdcecdb0)

```py
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/02323bb7-0c15-45e0-9a22-231415726e20)

```py
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats

sm.qqplot(df["Moderate Negative Skew"],line='45')

plt.show()
```
![image](https://github.com/user-attachments/assets/14cc701a-a00d-4ce1-b0f8-d89f0353f25d)

```py
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats

sm.qqplot(df["Moderate Negative Skew"],line='45')

plt.show()
```
![image](https://github.com/user-attachments/assets/5860aabd-7628-46ea-9508-52e4494b541b)

```py
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
```
![image](https://github.com/user-attachments/assets/40f91cbd-3dbd-45bb-8bf1-3e889d8a4df2)

```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/4206e0f0-897f-4ce9-87e7-a70ad0f4b3db)

# RESULT:
Hence performing Feature Encoding and Transformation process is Successful.

       
