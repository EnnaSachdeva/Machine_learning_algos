import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("/home/aadi-z640/machine_learning_algos/Datasets/winequality-red.csv")

    df["kfold"] = -1 # kfold

    df = df.sample(frac=1).reset_index(drop=True) # randomize data

    kf = model_selection.KFold(n_splits=5) # k=1

    for fold, (train_indx, test_indx) in enumerate(kf.split(X=df)):
        df.loc[test_indx, 'kfold'] = fold

    df.to_csv("/home/aadi-z640/machine_learning_algos/modified_datasets/k_fold_dataset.csv", index = False)

