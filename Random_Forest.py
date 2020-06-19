import pandas as pd 


df =pd.read_csv("Salary.csv")
print(df.head())

inputs = df.drop("salary" , axis = 'columns')
targets = df['salary']
#print(inputs)
#(targets)

from sklearn.preprocessing import LabelEncoder
le =  LabelEncoder()


inputs['company']  = le.fit_transform(inputs['company'])
inputs['degree']  = le.fit_transform(inputs['degree'])
inputs['job']  = le.fit_transform(inputs['job'])
#print(inputs)



# = inputs.drop(['company','job','degree'],axis = 'columns')
#print(input_n)

from sklearn.ensemble import RandomForestClassifier
reg = RandomForestClassifier(n_estimators = 100 , random_state = 0)
reg.fit(inputs,targets)

print("Score",reg.score(inputs, targets)*100)

from sklearn.externals import joblib
joblib.dump(reg, 'salary_prob')
mj = joblib.load('salary_prob')

ds = mj.predict([[1,3,1]])
print(ds)
if ds>=100000:
     print("Permanent Staff")
else:
     print("Temporary")
