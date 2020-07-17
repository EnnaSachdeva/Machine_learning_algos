import pandas as pd
from sklearn import model_selection
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

#%matplotlib inline

def create_folds(data):
    data["kfold"] = -1

    data = data.sample(frac=1).reset_index(drop=True) # randomize data

    # calculate the number of bins by sturge's rule
    num_bins = np.floor(1+np.log2(len(data)))

    data.loc[:,"bins"] = pd.cut(data["target"], bins=int(num_bins), labels=False)

    kf = model_selection.StratifiedKFold(n_splits=5)

    for fold, (train_indx, test_indx) in enumerate(kf.split(X=data, y = data.bins.values)):
        df.loc[test_indx, 'kfold'] = fold

    # drop the bins column
    data = data.drop("bins", axis=1)
    return data


if __name__ == "__main__":
    df = pd.read_csv("/home/aadi-z640/machine_learning_algos/Datasets/winequality-red.csv")

    b = sns.countplot(x="quality", data=df)

    b.set_xlabel("quality", fontsize=20)
    b.set_ylabel("count", fontsize=20)

    plt.show()

    # Distribution of labels
    x, y = datasets.make_regression(n_samples=15000, n_features=100, n_targets=1)

    df = pd.DataFrame(x, columns=[f"f_{i}" for i in range(x.shape[1])])

    df.loc[:, "target"] = y
    df = create_folds(df)


    '''
    df["kfold"] = -1 # kfold

    df = df.sample(frac=1).reset_index(drop=True) # randomize data

    targets = df.quality.values

    kf = model_selection.StratifiedKFold(n_splits=5) # k=1

    for fold, (train_indx, test_indx) in enumerate(kf.split(X=df, y = targets)):
        df.loc[test_indx, 'kfold'] = fold

    df.to_csv("/home/aadi-z640/machine_learning_algos/modified_datasets/stratified_k_fold_dataset.csv", index = False)
    '''

