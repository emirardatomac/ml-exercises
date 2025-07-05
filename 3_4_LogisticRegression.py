from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
  
df = pd.DataFrame(data = heart_disease.data.features)

df["target"] = heart_disease.data.targets

if df.isna().any().any():
    df.dropna(inplace=True)
    print("nan")
    
X = df.drop(["target"],axis=1).values
y = df.target.values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=42)

log_Reg = LogisticRegression(penalty="l2", C = 1 , solver="lbfgs",max_iter=100)
log_Reg.fit(X_train,y_train)

accuracy = log_Reg.score(X_test, y_test)
print("Logistic Regression Acc:" , accuracy)

