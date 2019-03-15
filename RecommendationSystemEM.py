import pandas as pd
import os
import numpy as np
from random import shuffle


class RecommendationSystem(object):

    def __init__(self,dataFolderPath:str):
        self.uidDict = dict()
        for row in pd.read_csv(os.path.join(dataFolderPath,"uid.csv"),
                               header=None).iterrows():
            self.uidDict[row[1][0]] = row[1][1]

        self.uidList: list = list(self.uidDict.keys())


        self.pidDict:dict = dict()
        for row in pd.read_csv(os.path.join(dataFolderPath,"pid.csv"),
                               header=None).iterrows():
            self.pidDict[row[1][0]] = row[1][1]


        self.pidList:list = list(self.pidDict.keys())
        self.uid_pid_explicit:dict = dict()

        for row in pd.read_csv(os.path.join(dataFolderPath,"explicit.csv"),
                               header=None).iterrows():
            self.uid_pid_explicit[int(row[1][0]),
                                  int(row[1][1])]=float(row[1][2])


        self.uid_pid_explicit_set:set = set(self.uid_pid_explicit.keys())

        self.uid_explicit=set(t[0] for t in list(self.uid_pid_explicit_set))
        self.pid_explicit=set(t[1] for t in list(self.uid_pid_explicit_set))


        self.uid_pid_implicit = set()
        for row in pd.read_csv(os.path.join(dataFolderPath,"implicit.csv"),
                               header=None).iterrows():
            self.uid_pid_implicit.add((int(row[1][0]),
                                       int(row[1][1])))
        self.uid_pid_implicit_backup = self.uid_pid_implicit

        self.uid_implicit=set(t[0] for t in list(self.uid_pid_implicit))
        self.pid_implicit=set(t[1] for t in list(self.uid_pid_implicit))


        self.latent_vars:int = None



        self.Nuid:int = len(self.uidDict) # which is m for MU in this case
        self.Npid:int = len(self.pidDict) # which is n for MI in this case
        # p mu u MU user
        # q mi i Mi item
        self.ran = False
        self.mu_result = None
        self.mi_result = None
        self.baseLineMatrix = None
        self.r_hat = None
    def get_users_similarity(self, uid1, uid2):

        explicit_tuples_pid1 = set([t[1] for t in self.uid_pid_explicit.keys() if t[0]==uid1])
        implicit_tuples_pid1 = set([t[1] for t in self.uid_pid_implicit if t[0]==uid1])
        tuples_pid1 = explicit_tuples_pid1|implicit_tuples_pid1
#         print(tuples_pid1)
        explicit_tuples_pid2 = set([t[1] for t in self.uid_pid_explicit.keys() if t[0]==uid2])
        implicit_tuples_pid2 = set([t[1] for t in self.uid_pid_implicit if t[0]==uid2])
        tuples_pid2 = explicit_tuples_pid2|implicit_tuples_pid2
#         print(tuples_pid2)
        return len(tuples_pid1&tuples_pid2)/len(tuples_pid1|tuples_pid2)



    def get_products_similarity(self, pid1, pid2):

        explicit_tuples_uid1 = set([t[0] for t in self.uid_pid_explicit.keys() if t[1]==pid1])
        implicit_tuples_uid1 = set([t[0] for t in self.uid_pid_implicit if t[1]==pid1])
        tuples_uid1 = explicit_tuples_uid1|implicit_tuples_uid1
#         print(tuples_uid1)
        explicit_tuples_uid2 = set([t[0] for t in self.uid_pid_explicit.keys() if t[1]==pid2])
        implicit_tuples_uid2 = set([t[0] for t in self.uid_pid_implicit if t[1]==pid2])
        tuples_uid2 = explicit_tuples_uid2|implicit_tuples_uid2
#         print(tuples_uid2)

        return len(tuples_uid1&tuples_uid2)/len(tuples_uid1|tuples_uid2)

    def matrixFactExplicitFeedback(self,
            latent_vars: int = 2,
            reg_strength: float = 0.0001,
            sgd_repetitions: int = 1000,
            sgd_learning_rate: float = 0.01):

        # here even the parameter name is trainint_points
        # normally you only pass in a point in a list
        # such as [(1,2)]
        def target_fun(currMu_input:np.matrixlib.defmatrix.matrix,
                       currMi_input:np.matrixlib.defmatrix.matrix,
                       training_points):
            target = 0
            for uid_pid_pair in training_points:
                r_ui = self.uid_pid_explicit.get(uid_pid_pair)
                curr_uid = uid_pid_pair[0]
                curr_pid = uid_pid_pair[1]

                curr_uid_matrix_index = self.uidList.index(curr_uid)
                curr_pid_matrix_index = self.pidList.index(curr_pid)

                pu = currMu_input[:,curr_uid_matrix_index]
                qi = currMi_input[:,curr_pid_matrix_index]
                pu_norm = np.linalg.norm(pu,2)
                qi_norm = np.linalg.norm(qi,2)
                reg = reg_strength*(pu_norm**2+qi_norm**2)
                qiTpu = (np.transpose(qi)*pu)[0,0]
                target += r_ui**2 - qiTpu + reg
            return target


        def get_next_step_gradient(currMu_input,
                                   currMi_input,
                                   training_points):
            currMu_tmp = currMu_input.copy()
            currMi_tmp = currMi_input.copy()
            for uid_pid_pair in training_points:

                r_ui = self.uid_pid_explicit.get(uid_pid_pair)
                curr_uid = uid_pid_pair[0]
                curr_pid = uid_pid_pair[1]

                curr_uid_matrix_index = self.uidList.index(curr_uid)
                curr_pid_matrix_index = self.pidList.index(curr_pid)

                pu = currMu_tmp[:,curr_uid_matrix_index]
                qi = currMi_tmp[:,curr_pid_matrix_index]
                qiTpu = (np.transpose(qi)*pu)[0,0]
                e_ui = r_ui - qiTpu

                new_qi = qi + sgd_learning_rate*(e_ui*pu - reg_strength*qi)
                new_pu = pu + sgd_learning_rate*(e_ui*qi - reg_strength*pu)
                currMu_tmp[:,curr_uid_matrix_index] = new_pu
                currMi_tmp[:,curr_pid_matrix_index] = new_qi

            return currMu_tmp, currMi_tmp




        self.latent_vars = latent_vars
        random_shuffled_keys = list(self.uid_pid_explicit.keys())
        shuffle(random_shuffled_keys)

        currMu = np.matrix(np.ones([latent_vars, self.Nuid])*1)
        currMi = np.matrix(np.ones([latent_vars, self.Npid])*1)

        for currIter in range(sgd_repetitions):
            for uid_pid_pair in random_shuffled_keys:
                currMu,currMi=get_next_step_gradient(currMu,
                                                     currMi,
                                                     [uid_pid_pair])



        return currMu, currMi


    def run(self,sim_thresh:float = 0.1, testCase:int =0):

        if self.ran:
            print("ran, check result")
            return

        # initialization

        cutOff = lambda x, thresh: (x>=thresh)*x




        def q_Tp_(qVarI, pVarU, mu, mi):
                curr_uid_matrix_index = self.uidList.index(pVarU)
                curr_pid_matrix_index = self.pidList.index(qVarI)

                pu = mu[:,curr_uid_matrix_index]
                qi = mi[:,curr_pid_matrix_index]
                return (np.transpose(qi)*pu)[0,0]


        uid_pid_explicit_hat = dict()
        training_dict = self.uid_pid_explicit.copy()
        currEffictive = True
        mu = None
        mi = None

        if testCase == 1:
            # print("directly outputting the initialization")
            mu, mi = self.matrixFactExplicitFeedback()
            self.mu_result = mu.copy()
            # print(mu)
            self.mi_result = mi.copy()
            # print(mi)
            return mu, mi

        tmp_uid_pid_implicit = self.uid_pid_implicit.copy()

        mu, mi = self.matrixFactExplicitFeedback()
        while currEffictive:
            # print("currently")


            for uid_pid_pair_i in self.uid_pid_implicit:
                curr_uid = uid_pid_pair_i[0]
                curr_pid = uid_pid_pair_i[1]

                has_uid = lambda m: sum([k[0] == m for k in list(training_dict.keys())]) > 0
                has_pid = lambda m: sum([k[1] == m for k in list(training_dict.keys())]) > 0

                if uid_pid_pair_i in training_dict and (testCase == 2 or testCase == 0):
                    # print("case 1 running")
                    uid_pid_explicit_hat[uid_pid_pair_i] = q_Tp_(curr_pid,curr_uid,mu,mi)
                    tmp_uid_pid_implicit.remove(uid_pid_pair_i)
                    continue

                if has_uid(curr_uid) and not has_pid(curr_pid) and (testCase==3 or testCase==0 or testCase==5):
                    # print("case 2 running")
                    up = sum([ cutOff(self.get_products_similarity(curr_pid,j),sim_thresh)*q_Tp_(j,curr_uid, mu, mi) for j in self.pidList if j is not curr_pid])
                    down = sum([cutOff(self.get_products_similarity(curr_pid,j),sim_thresh) for j in self.pidList if j is not curr_pid])
                    tmp_uid_pid_implicit.remove(uid_pid_pair_i)
                    uid_pid_explicit_hat[uid_pid_pair_i] = up/down
                    continue

                if not has_uid(curr_uid) and has_pid(curr_pid) and (testCase==4 or testCase==0 or testCase==5):
                    # print("case 3 running")
                    up = sum([cutOff(self.get_users_similarity(curr_uid,v), sim_thresh)*q_Tp_(curr_pid,v,mu,mi) for v in self.uidList if v is not curr_uid])
                    down = sum([cutOff(self.get_users_similarity(curr_uid, v),sim_thresh) for v in self.uidList if v is not curr_uid])
                    tmp_uid_pid_implicit.remove(uid_pid_pair_i)
                    uid_pid_explicit_hat[uid_pid_pair_i] = up/down
                else:
                    # print("doing nothing")
                    currEffictive = False

            previousEst = np.transpose(mu)*mi
            mu, mi = self.matrixFactExplicitFeedback()
            currEst = np.transpose(mu)*mi

            if len(tmp_uid_pid_implicit) == 0 and abs((currEst - previousEst).sum())<0.005:
                currEffictive = False

            for t in uid_pid_explicit_hat:
                if t not in training_dict:
                    training_dict[t] = uid_pid_explicit_hat[t]

        self.mu_result = mu.copy()
        # print(mu)
        self.mi_result = mi.copy()
        # print(mi)
        return mu, mi


    def calculateRMSE(self, baseLineFile):
        r_hat = np.transpose(self.mu_result)*self.mi_result
        baseLineDf = pd.read_csv(baseLineFile, header=None)
        mat_num_row = len(self.uidList)
        mat_num_col = len(self.pidList)

        baseLineMatrix = np.matrix(np.zeros( (mat_num_row,mat_num_col) ))

        for curr_row in baseLineDf.iterrows():
            curr_uid = int(curr_row[1][0])
            curr_pid = int(curr_row[1][1])
            curr_rating = float(curr_row[1][2])

            curr_uid_loc = self.uidList.index(curr_uid)
            curr_pid_loc = self.pidList.index(curr_pid)
            baseLineMatrix[curr_uid_loc, curr_pid_loc] = curr_rating

        self.r_hat = r_hat.copy()
        self.baseLineMatrix = baseLineMatrix.copy()
        squaredError = (np.power((baseLineMatrix-r_hat), 2))
        RMSE = np.sqrt(squaredError.sum()/squaredError.size)
        return RMSE



for i in range(0,6):
    recommender = RecommendationSystem(dataFolderPath="xsmall_data/")
    runResult = recommender.run(sim_thresh = 0.1, testCase = i)
    result = recommender.calculateRMSE("xsmall_data/ratings.csv")
    print(result)
