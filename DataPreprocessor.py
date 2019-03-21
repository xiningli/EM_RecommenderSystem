import pandas as pd
from sklearn.model_selection import train_test_split
import os


def preprocess(inFile, outFolder, explicit_ratio: float):

    origDat = pd.read_csv(inFile, header=None)
    train, test = train_test_split(origDat, test_size=1-explicit_ratio)

    test = test.drop(test.columns[-1], axis=1)
    # print("saving for the implicit file")
    test.to_csv(os.path.join(outFolder, "implicit.csv"), header=False, index=False)
    # print("saving for the explicit file")
    train.to_csv(os.path.join(outFolder, "explicit.csv"), header=False, index=False)



preprocess("xsmall_data/ratings.csv", "xsmall_data/", explicit_ratio=0.2)
