import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import ast

# data_dict = pickle.load(open('data.pickel', 'rb'))
# print(data_dict)
# df = pd.DataFrame({'data': data_dict['data'], 'labels': data_dict['labels']})
# df.to_csv(r'file.csv', index=False)

data = pd.read_csv('file.csv')

data['data'] = data['data'].apply(ast.literal_eval)
data['data'] = data['data'].apply(np.array)

data['labels'].unique()

le = LabelEncoder()
data['labels'] = le.fit_transform(data['labels'])
data['length'] = data['data'].apply(len)

data['length'].unique()

print(data.head())

max_length = data['length'].max()
data = data[data['length'] != max_length]
max_length = data['length'].max()

data['data'] = data['data'].apply(lambda x: np.pad(x, (0, max_length - len(x))))

X = np.stack(data['data'].values)
y = data['labels'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)

print(clf.score(X_test, y_test))

f = open('model.p','wb')
pickle.dump({'model' : clf}, f)
f.close()
