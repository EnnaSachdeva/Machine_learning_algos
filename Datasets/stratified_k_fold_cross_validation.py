import pandas as pd
from sklearn import tree
from sklearn import metrics

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import model_selection


df = pd.read_csv("/home/enna/Desktop/machine_learning_algos/Datasets/winequality-red.csv")

# wine quality is a regression problem/ classification problem: quality is a real number between 0 to 10
quality_mapping = {
    3: 0,
    4: 1,
    5: 2,
    6: 3,
    7: 4,
    8: 5
}



df.loc[:, "quality"] = df.quality.map(quality_mapping)

df["kfold"] = -1

# splitting the data into training and testing frames
df = df.sample(frac=1).reset_index(drop=True) # shuffle the data

y = df.target.values

kf = model_selection.StratifiedKFold(n_splits=5)


# fill the new kfold column
for t, (t_, v_) in enumerate(kf.split())
df_train = df.head(1000) # take the first 1000 as training data
df_test = df.tail(599) # testing data

train_accuracies = [0.5]
test_accuracies = [0.5]


for depth in range(1, 25):

    # implement decision tree classifier for various depths
    clf = tree.DecisionTreeClassifier(max_depth=depth)

    input_features = df_train.loc[:, ~df_train.columns.isin(['quality'])]

    clf.fit(input_features, df_train.quality) # fit the model

    train_predictions = clf.predict(input_features)

    test_predictions = clf.predict(df_test.loc[:, ~df_test.columns.isin(['quality'])])

    train_accuracy = metrics.accuracy_score(df_train.quality, train_predictions)

    test_accuracy = metrics.accuracy_score(df_test.quality, test_predictions)

    print("Train accuracy: ", train_accuracy, "Train accuracy: ", test_accuracy)

    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)



plt.figure(figsize = (10, 5))
sns.set_style("whitegrid")
plt.plot(train_accuracies, label = "train_accuracy")
plt.plot(test_accuracies, label = "test_accuracy")
plt.legend(loc = "upper left", prop = {'size' : 15})
plt.xticks = (range(0, 26, 5))
plt.xlabel("max_depth", size = 20)
plt.ylabel("accuracy", size = 20)
plt.show()







