import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lazypredict.Supervised import LazyClassifier
import pickle

data = pd.read_csv("diabetes (1).csv")
# data split
target = "Outcome"
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)


# data preprocessing
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# train model

# # model = SVC()
# model = SVC()
# model = model.fit(x_train, y_train)
# y_predict = model.predict(x_test)
#
# prediction = classification_report(y_test, y_predict)
# #               precision    recall  f1-score   support
# #
# #            0       0.77      0.83      0.80        99
# #            1       0.65      0.56      0.60        55
# #
# #     accuracy                           0.73       154
# #    macro avg       0.71      0.70      0.70       154
# # weighted avg       0.73      0.73      0.73       154


# # model = decisiontree
model = DecisionTreeClassifier(random_state=100, criterion="gini")
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
prediction = classification_report(y_test, y_predict)
print(prediction)

# prediction = classification_report(y_test, y_predict)
#
#               precision    recall  f1-score   support
#
#            0       0.84      0.77      0.80        99
#            1       0.64      0.75      0.69        55
#
#     accuracy                           0.76       154
#    macro avg       0.74      0.76      0.75       154
# weighted avg       0.77      0.76      0.76       154

# # model = randomforest
# model = RandomForestClassifier(n_estimators=100 ,random_state=100, verbose=2, criterion="entropy")
# model.fit(x_train, y_train)
# y_predict = model.predict(x_test)
#
# prediction = classification_report(y_test, y_predict)
# print(prediction)

# # GridSearchCV
# params = {
#     "n_estimators" : [100, 200, 300],
#     "criterion" : ["gini", "entropy", "log_loss"],
#
# }
#
# grid_search = GridSearchCV(param_grid= params,estimator = RandomForestClassifier(random_state=100) ,scoring= "recall", cv=5, verbose=2)
# grid_search.fit(x_train, y_train)
# y_predict = grid_search.predict(x_test)
#
# print(grid_search.best_params_)
# print(grid_search.best_score_)
# print(grid_search.best_estimator_)
# prediction = classification_report(y_test, y_predict)
# print(prediction)

# {'criterion': 'entropy', 'n_estimators': 100}
# 0.7785419165667067
# RandomForestClassifier(criterion='entropy', random_state=100)
#               precision    recall  f1-score   support
#
#            0       0.80      0.81      0.80        99
#            1       0.65      0.64      0.64        55
#
#     accuracy                           0.75       154
#    macro avg       0.72      0.72      0.72       154
# weighted avg       0.75      0.75      0.75       154

clf= LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
lazy_model, prediction = clf.fit(x_train, x_test, y_train, y_test)

with open("diabetes_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)



