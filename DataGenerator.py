import numpy as np
from random import shuffle
import os

def generate(Nuser,
             Nitem,
             explicit_dist,
             implicit_dist,
             e_ratio,
             outdir):
    Nrating = Nuser * Nitem
    eiBoolList = []

    userStr = ""
    for u in range(Nuser):
        userStr += str(u) + "," + str(u) + "\n"
    itemStr = ""
    for i in range(Nitem):
        itemStr += str(i) + "," + str(i) + "\n"

    with open(os.path.join(outdir, "uid.csv"), 'w') as uout:
        uout.write(userStr)

    with open(os.path.join(outdir, "pid.csv"), 'w') as uout:
        uout.write(itemStr)


    for j in range(Nrating):
        eiBoolList.append(False)

    for j in range(int(Nrating*e_ratio)):
        eiBoolList[j] = True
    shuffle(eiBoolList)
    # return eiBoolList
    resultList = []
    for obs in eiBoolList:
        if obs == True:
            tmp = explicit_dist()
            while tmp<0:
                tmp = explicit_dist()
        else:
            tmp = implicit_dist()
            while tmp<0:
                tmp = implicit_dist()

        resultList.append(tmp)

    outExplicit = ""
    outImplicit = ""
    outRating = ""

    resultMatrix = np.zeros((Nuser,Nitem))

    ticker = 0
    indexMatrix = np.zeros((Nuser, Nitem))
    for i in range(Nuser):
        for j in range(Nitem):
            indexMatrix[i,j] = ticker
            ticker += 1


    for i in range(Nuser):
        for j in range(Nitem):

            tmpRating = resultList[int(indexMatrix[i,j])]
            resultMatrix[i,j] = tmpRating
            if eiBoolList[int(indexMatrix[i,j])]==True:
                outExplicit += str(i)+","+str(j)+ "," + str(tmpRating) + "\n"
            else:
                outImplicit+= str(i)+","+str(j) + "\n"
            outRating += str(i) + ","+str(j) + "," + str(tmpRating) + "\n"

    with open(os.path.join(outdir,"explicit.csv"), 'w') as oute:
        oute.write(outExplicit)
    with open(os.path.join(outdir,"implicit.csv"), 'w') as outi:
        outi.write(outImplicit)
    with open(os.path.join(outdir,"ratings.csv"), 'w') as outr:
        outr.write(outRating)

    return outExplicit,outImplicit


