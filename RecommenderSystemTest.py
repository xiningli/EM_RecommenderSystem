from RecommendationSystemEM import RecommendationSystem
from DataPreprocessor import preprocess
import numpy as np

for j in range(100):
    s="PerfectData"
    for i in range(0,2):
        preprocess(s+"/ratings.csv", s+"/", explicit_ratio=0.8)
        recommender = RecommendationSystem(dataFolderPath=s+"/")
        runResult = recommender.run(sim_thresh = 0.1, testCase = i)
        result = recommender.calculateRMSE(s+"/ratings.csv")
        print(result, end="")
        if i<1:
            print(",", end="")
    print("\n", end="")
