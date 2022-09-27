import time
from CoLinFactorUCB_utils import *


class ColinConstruct:
    def __init__(self, itemFeatureDim, lambd, alpha, W, initializeBanditParams='zeros'):
        # initializeBanditParams can also be random for random bandit parameters initializations, by default it's zeros

        # Letting the Length of adjacency matrix to be the userNum for local purposes
        # This actualLy stands for number of UNIQUE users, aka, number of clusters
        self.userNum = len(W)
        self.W = W
        self.alpha = alpha
        self.itemDim = itemFeatureDim
        self.lambd = lambd

        # I don't need to keep track of A, only need A inverse
        # will be updated with gathered information, wilL always be kept inverted
        self.AInv = (1 / self.lambd) * np.identity(self.itemDim * self.userNum)
        self.b = np.zeros(self.itemDim * self.userNum)  # will be updated with gathered information

        # The Colin algorithm in the original paper has Line 11, a \hat{v},
        # that is the vectorized version of userTheta beLow.
        # We can just always get the \hat{v} and matrixize it, then we don't need a \hat{v}
        # To be Learned, uppercase \hat(\theta} in the paper, CoLin alg Line 6 has it in the UCB term, first term
        self.userTheta = np.zeros((self.itemDim, self.userNum))

        if initializeBanditParams == 'random':
            self.userTheta = l1NormalizationByCol(np.random.rand(self.itemDim, self.userNum))

        # (bar(\Theta) in the paper, combination of user similarities with item-user preferences,
        # again Line 6, the term inside the vec(.)
        self.barTheta = np.zeros((self.itemDim, self.userNum))

        # since this is static, we just fix the kronecker multiplication of W and Identity matrix of proper size
        self.kronnedW = np.kron(np.transpose(W), np.identity(self.itemDim))

        # Confidence Bound - Optimistic part, CEO (paper: matrix C), second term of the UCB term on Line 6
        self.CBO = np.dot(np.dot(self.kronnedW, self.AInv), np.transpose(self.kronnedW))

    def updateParameters(self, itemContext, reward, userID, normalizeBanditParams=False):
        # By default, the bandit parameters are not normalized by their L1 norms,
        # a False boolean is required as the fourth argument to not normalize the parameters
        # The first element of the second term in the right-hand side of line 8,
        # the X and X transpose is the update on A
        X = vectorize(np.outer(itemContext, self.W.T[userID]))

        # Since doing full inverse takes longer we do a rank-one update.
        # Comparing different approaches in terms of execution time, we use the below lines,
        # where the discussion can be found on
        # https://timvieira.github.io/blog/post/2021/03/25/fast-rank-one-updates-to-matrix-inverse/
        # Also saved on wayback machine, in case it disappears
        # Relevant code is at https://timvieira.github.io/blog/post/2021/03/25/fast-rank-one-updates-to-matrix-inverse/
        interim = Inv(self.AInv)
        interim.rank_one_update(X, X)
        self.AInv = interim.value

        # Line 9 in CoLin
        self.b = reward * X

        if normalizeBanditParams:
            # Normalizing the bandit parameters of each user/cluster
            self.userTheta = l1NormalizationByCol(matrixize(np.dot(self.AInv, self.b), len(itemContext)))
        else:
            self.userTheta = matrixize(np.dot(self.AInv, self.b), len(itemContext))

        # Recall that this is combination of user similarities with other user preferences,
        # Line 6, the term inside the vec(.)
        # If one is comparing bandit parameters across different algorithms, one should use this
        # as the userTheta bandit parameter is the true bandit parameter for each user, if they were in a 'vacuum
        # We use this in UCB calculations as the first term in UCB is arm context combined with
        # bandit parameters with other users' effects plugged in
        self.barTheta = np.dot(self.userTheta, self.W)

        # Update the confidence term in the UCB
        # This method also return the time it takes to calculate the CBO matrix, that can be used for a sanity check
        stime = time.time()
        self.CBO = np.dot(np.dot(self.kronnedW, self.AInv), np.transpose(self.kronnedW))
        etime = time.time()
        return etime - stime

    def getRec(self, items_info, userID):
        # Instead of calculating the UCB for each arm and Looping over that calculation, here we do it in one go.
        # Just get the rows of a column-vector to denote the UCB of each arm.
        # Here the assumption on the data is as follows:
        # Each row stores one item, the first column stores the item ID (not used within this method),
        # the other columns (there should be self.itemDim many of them) store the contextual information of that item.
        meanVector = np.dot(self.barTheta.T[userID], items_info.T[1:, :])

        # This part calculates the UCB of each item and returns an array where the UCB of i-th item in items info is
        # in the i-th position of the returned array
        item_temp_features = np.empty([len(items_info), self.itemDim * self.userNum])
        for i in range(len(items_info)):
            TempFeatureMean = np.zeros((self.itemDim, self.userNum))
            TempFeatureMean.T[userID] = items_info[i, 1:]
            item_temp_features[i, :] = vectorize(TempFeatureMean)
        TempFeatureVariance = np.sqrt(np.dot(np.dot(item_temp_features, self.CBO), item_temp_features.T))
        UCBs = meanVector + self.alpha * np.diag(TempFeatureVariance)

        # Sort the UCBs in decreasing order
        item_positions = np.argsort(UCBs)[::-1]
        # Reorder the rows in item info to reflect the ordering of the UCBs
        items = []
        for i in range(len(items_info)):
            items.append(items_info[item_positions[i]])
        # Return the items, this allows for multiple selections in one iteration
        return items

    def return_userTheta_AInv(self):
        # returns the userTheta and AInv, these two are enough to warm start
        # another instance of the same user-relation set
        return {'userThetas': self.userTheta.T, 'AInv': self.AInv}
        # CoLinStruct returns the userTheta transposed because that puts the users on the rows,
        # makes it easier to compare parameters
        # userTheta is read as transposed in Simulator, so if it is not transposed here, it should be fixed in Simulator

    def fitParams(self, userTheta, A_inverse_in=False, AInv=np.empty(1)):
        # Assumes that the userTheto and AlNv has the correct structure and dimensions
        # By default, A inverse is not assumed to be plugged in
        # If only userTheta is provided, then only userTheta is plugged in
        # To plug in A inverse as well, the second argument should be True and
        # the third argument should be the correct AInv
        self.userTheta = userTheta
        if A_inverse_in:
            self.AInv = AInv

    def warmStart(self, both=False):
        # Can be called after properly provided userTheta (and AInv) is injected to the object
        # By default, this assumes only userTheta is injected and hence only barTheta is updated
        # If both user Theta and AInv are injected, then the only argument should be True
        self.barTheta = np.dot(self.userTheta, self.W)
        if both:
            self.CBO = np.dot(np.dot(self.kronnedW, self.AInv), np.transpose(self.kronnedW))
            self.b = np.dot(np.linalg.inv(self.AInv), vectorize(self.userTheta))
            # here b is using the regular inverse since we do not have access to A

    def readParams(self, warmStartPaths):
        path_userTheta = str(warmStartPaths['userTheta'])
        userTheta = np.transpose(pd.read_csv(path_userTheta, header=None, sep=",").to_numpy())
        # here userTheta is transposed because CoLinStruct returns the userTheta transposed
        # both should be updated if one is updated
        # if we have a path for AInv, we wilL note it, and we will warm start accordingly
        # otherwise, we wiLl onLy fit userTheta and warm start onLy with it

        if warmStartPaths['AInv_in']:
            path_AInv = str(warmStartPaths['AInv'])
            AInv = pd.read_csv(path_AInv, header=None, sep=",").to_numpy()
            self.fitParams(userTheta, True, AInv)
            # second argument is True because we are reading AInv as well
            self.warmStart(True)
        else:
            self.fitParams(userTheta)
            self.warmStart()
