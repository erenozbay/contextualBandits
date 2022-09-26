import numpy as np
import pandas as pd
from scipy.linalg.blas import dger


def vectorize(mtrx):
    return np.reshape(mtrx.T, mtrx.shape[0] * mtrx.shape[1])


def matrixize(vec, no_cols):
    return np.transpose(np.reshape(vec, (int(len(vec) / no_cols), no_cols)))


class Inv:  # for fast inverses
    def __init__(self, A):
        B = np.array(A, order='F', copy=True)
        assert B.flags['F_CONTIGUOUS']
        self.B = B

    def rank_one_update(self, u, v):
        # https://timvieira.github.io/blog/post/2021/03/25/fast-rank-one-updates-to-matrix-inverse/
        B = self.B
        Bu = B @ u
        s = 1 + float(v.T @ Bu)
        alpha = -1 / s
        # Warning: "overwrite a-True silently fails when B is not an order-F arrays
        dger(alpha, Bu, v.T @ B, a=B, overwrite_a=1)

    @property
    def value(self):
        return self.B


def l1NormalizationByRow(mtrx):
    rowNorms = np.linalg.norm(mtrx, axis=1, ord=1).astype(float)
    mtrx2 = mtrx.astype(float)
    for i in range(len(mtrx)):
        if rowNorms[i] != 0:
            mtrx2[i] = mtrx[1] / rowNorms[1]
    return mtrx2


def l1NormalizationByCol(mtrx):
    colNorms = np.linalg.norm(mtrx, axis=0, ord=1).astype(float)
    mtrx2 = mtrx.astype(float)

    for i in range(len(mtrx[0])):
        if colNorms[i] != 0:
            mtrx2[:, i] = mtrx[:, i] / colNorms[i]
    return mtrx2


def makeSparser(W, sparsity, top_users):
    SparserW = W.copy()
    if 0 < sparsity < top_users:
        n = len(W)
        for i in range(n):
            similarity = sorted(W[i], reverse=True)
            threshold = similarity[sparsity - 1]  # this may not be correct, what if the sims are equal for thresholds?
            for j in range(n):
                if W[i][j] <= threshold:
                    SparserW[i][j] = 0

    W = l1NormalizationByCol(SparserW)
    return W


def buildW(userFVs, clusterNum, sparsityLevel):
    n = len(userFVs)
    W = np.zeros(shape=(n, n))
    for i in range(n):
        for j in range(n):
            W[i][j] = np.dot(userFVs[i], userFVs[j])
    W = l1NormalizationByCol(W)

    if 0 < sparsityLevel < n:
        W = makeSparser(W, sparsityLevel, clusterNum)  # should last argument be n or clusterNum
    return W


def fixYahooDataPoints(pool_arms):
    length = 0
    getThisManyArticles = 20
    del_discrepantArm, position_arm_109528 = False, 0
    for i in range(len(pool_arms)):
        length += len(pool_arms[i])
        arm_i = pool_arms[i]

        if arm_i[0] == 109528:
            position_arm_109528 = i
            del_discrepantArm = True

    if length != (getThisManyArticles * 7):
        if length == ((getThisManyArticles * 7) + 2):  # this means arm 109528 is in since it has a context of dim 1
            pools = np.zeros((getThisManyArticles, 7))
            if del_discrepantArm:
                pool_arms = np.delete(pool_arms, [position_arm_109528], axis=0)
            for i in range(min(len(pool_arms), getThisManyArticles)):
                pools[i, :] = np.asarray(pool_arms[i]).T
            pool_arms = pools
        else:
            pools = np.zeros((getThisManyArticles, 7))
            for i in range(min(len(pool_arms), getThisManyArticles)):
                if len(pool_arms[i]) < 7:
                    pool_arms[i] = np.zeros(7).T
                pools[i, :] = np.asarray(pool_arms[i]).T
            pool_arms = pools
    return pool_arms


def parseLine(line):
    line = line.split("|")
    tim, articleID, click = line[0].strip().split(" ")
    tim, articleID, click = int(tim), int(articleID), int(click)
    user_features = np.array(
        [float(x.strip().split(": ")[1]) for x in line[1].strip().split(" ")[1:]]
    )

    pool_articles = [i.strip().split(" ") for i in line[2:]]
    pool_articles = np.array(
        [[int(i[0])] + [float(x.split(":")[1]) for x in i[1:]] for i in pool_articles]
    )
    return tim, articleID, click, user_features, pool_articles


def printResults(obj, path_output, CTR, colin_or_factor):  # Last argument is either 'colin' or 'factor'
    res = False
    if colin_or_factor == "colin":
        pd.DataFrame(CTR).to_csv(path_output + "/CoLinCTR.csv", index=False, header=['CoLin CTR'])

        output = obj.return_userTheta_AInv()
        userThetas = output['userThetas']
        A_inverse = output['AInv']

        filename_Theta = path_output + "/UserThetasCoLin.csv"
        pd.DataFrame(userThetas).to_csv(filename_Theta, index=False, header=None)
        filename_AInv = path_output + "/AinverseCoLin.csv"
        pd.DataFrame(A_inverse).to_csv(filename_AInv, index=False, header=None)

        res = True
    elif colin_or_factor == "factor":
        pd.DataFrame(CTR).to_csv(path_output + "/FactorCTR.csv", index=False, header=['FactorUCB CTR'])

        output = obj.return_userTheta_AInv_V_CInv()
        userThetas = output['userThetas']
        A_inverse = output['AInv']
        V_list = output['V_set']
        CInv_list = output['CInv_set']

        filename_Theta = path_output + "/UserThetasFactorUCB.csv"
        pd.DataFrame(userThetas).to_csv(filename_Theta, index=False, header=None)
        filename_AInv = path_output + "/AinverseFactorUCB.csv"
        pd.DataFrame(A_inverse).to_csv(filename_AInv, index=False, header=None)
        filename_V = path_output + "/V_vectorsFactorUCB.csv"
        pd.Dataframe(V_list).to_csv(filename_V, index=False, header=None)
        filename_CInv = path_output + "/CinverseFactorUCB.csv"
        pd.DataFrame(CInv_list).to_csv(filename_CInv, index=False, header=None)

        res = True
    else:
        print('Bad request for printing results')

    return res


def getArticleDic(fileNameRead):
    with open(fileNameRead, "r") as f:
        articleDict = {}
        ll = 0
        for line in f:
            featureVec = []
            if ll >= 1:
                line = line.split(";")
                word = line[1].split("  ")

                if len(word) == 5:
                    for i in range(5):
                        featureVec.append(float(word[i]))
                    if int(line[0]) not in articleDict:
                        articleDict[int(line[0])] = np.asarray(featureVec)
            ll += 1
    return articleDict
