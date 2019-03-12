from RecommendationSystemEM import RecommendationSystem
import numpy as np
rms = RecommendationSystem(dataFolderPath="toyData")

runResult = rms.run(sim_thresh = 0.1, testCase = 1)

print(runResult)

print(np.transpose(runResult[0])*runResult[1])
