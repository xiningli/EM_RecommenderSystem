from RecommendationSystemEM import RecommendationSystem
from DataPreprocessor import preprocess
from DataGenerator import generate
import numpy as np


for j in range(20):
    s="simulationByDistribution"
    generate(5,
               4,
               lambda : 0.7* np.random.normal(3, 2) + 0.3*np.random.normal(4, 0.1),
               lambda : np.random.normal(3, 3),
               0.8,
               s)

    for i in range(0,2):

        preprocess(s+"/ratings.csv", s+"/", explicit_ratio=0.8)
        recommender = RecommendationSystem(dataFolderPath=s+"/")
        runResult = recommender.run(sim_thresh = 0.1, testCase = i)
        result = recommender.calculateRMSE(s+"/ratings.csv")
        print(result, end="")
        if i<1:
            print(",", end="")
    print("\n", end="")
