import pandas as pd  # Reading parquets
from imblearn.over_sampling import SMOTE
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler

train = pd.read_parquet("train.parquet.gzip")
test = pd.read_parquet("test.parquet.gzip")
org_test = test

# print(train.shape)

print('No Frauds', round(train['Class'].value_counts()[0] / len(train) * 100, 2), '% of the dataset')
print('Frauds', round(train['Class'].value_counts()[1] / len(train) * 100, 2), '% of the dataset')

# Note: Notice how imbalanced is our original dataset! Most of the transactions are non-fraud.
# If we use this dataframe as the base for our predictive models and analysis we might get a lot of errors and our algorithms
# will probably overfit since it will "assume" that most transactions are not fraud. But we don't want our model to assume,
# we want our model to detect patterns that give signs of fraud!

# Robust scalar will fit for outliers other scalar won't fit for outliers
rob_scaler = RobustScaler()

# train['Amount'].values will give output values as row format so reshaping to column format
train['scaled_amount'] = rob_scaler.fit_transform(train['Amount'].values.reshape(-1, 1))
train['scaled_time'] = rob_scaler.fit_transform(train['Time'].values.reshape(-1, 1))

test['scaled_amount'] = rob_scaler.fit_transform(test['Amount'].values.reshape(-1, 1))
test['scaled_time'] = rob_scaler.fit_transform(test['Time'].values.reshape(-1, 1))

# Checking whether any null values.
print(train.isnull().sum().max())

train.drop(['Time', 'Amount'], axis=1, inplace=True)
test.drop(['Time', 'Amount'], axis=1, inplace=True)

scaled_amount = train['scaled_amount']
scaled_time = train['scaled_time']

test_scaled_amount = test['scaled_amount']
test_scaled_time = test['scaled_time']

train.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
train.insert(0, 'scaled_amount', scaled_amount)
train.insert(1, 'scaled_time', scaled_time)

test.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
test.insert(0, 'scaled_amount', test_scaled_amount)
test.insert(1, 'scaled_time', test_scaled_time)

# Amount and Time are Scaled!

print(train.head())

print('No Frauds', round(train['Class'].value_counts()[0] / len(train) * 100, 2), '% of the dataset')
print('Frauds', round(train['Class'].value_counts()[1] / len(train) * 100, 2), '% of the dataset')

X = train.drop('Class', axis=1)
y = train['Class']

sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)

for train_index, test_index in sss.split(X, y):
    print("Train:", train_index, "Test:", test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Turn into an array
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

# Keras || OverSampling (SMOTE):
# Synthetic Minority Over-sampling TEchnique - It will create the new point between minority values within the minority range.
sm = SMOTE(random_state=42)
Xsm_train, ysm_train = sm.fit_sample(X_train, y_train)

n_inputs = Xsm_train.shape[1]
print('model input=', n_inputs)
model = Sequential([
    Dense(n_inputs, input_shape=(n_inputs,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(10, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(Adam(lr=0.002), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

model.fit(Xsm_train, ysm_train, validation_split=0.2, batch_size=300, epochs=20, shuffle=True, verbose=2)

fraud_predictions = model.predict_classes(X_test, batch_size=200, verbose=0)

smote = confusion_matrix(y_test, fraud_predictions)

print("Confusion Matrix")
print(smote)

print("Accuracy")
print(accuracy_score(y_test, fraud_predictions))

test_pred = model.predict_classes(test, batch_size=200, verbose=0)

org_test['class'] = test_pred

org_test.to_csv('test_predicted.csv')

print(recall_score(y_test, fraud_predictions))
print(precision_score(y_test, fraud_predictions))
print(f1_score(y_test, fraud_predictions))
