import time
from CoLinFactorUCB_utils import *


class FactorUCBArmStruct:
    def __init__(self, ID, itemFeatureDim, latent_dim, lambd):
        self.itemID = ID
        self.itemFeatureDim = itemFeatureDim
        dim = itemFeatureDim + latent_dim

        # same with AInv, only need inverse of the C matrix initialized in Line 7
        self.CInv = (1 / lambd) * np.identity(latent_dim)
        self.d = np.zeros(latent_dim)
        self.count = {}

        # V vector contains the actual contextual information of arms as well as the latent features
        # In FactorUCB algorithm, this takes the place of, e.g., (x_a, \hat{v}_{a,t}) in the mean term of UCB
        self.V = np.abs(np.random.rand(dim))
        self.count = 0

    def updateParameters(self, barTheta, click, userID):
        # rank-one update in line 14, barTheta is the userTheta combined with W matrix, so this is correct
        X = barTheta.T[userID][self.itemFeatureDim:]
        interim = Inv(self.CInv)
        interim.rank_one_update(X, X)
        self.CInv = interim.value
        self.d += X * (click - barTheta.T[userID][:self.itemFeatureDim].dot(self.V[:self.itemFeatureDim]))

        # the relevant part (latent part) of the V vector is updated with CInv and d, Line 16 in the algorithm
        self.V[self.itemFeatureDim:] = np.dot(self.CInv, self.d)

    def contextsIn(self, single_arm_info):
        self.V[:self.itemFeatureDim] = single_arm_info
        self.count = 1  # just to make sure I don't plug in contexts again and again

    # this part should be different, this is not too good
    # just here to make sure that I don't plug in contexts again and again
    def checkv(self):
        return self.count == 0


class FactorUCBUserStruct:
    def __init__(self, itemFeatureDim, latent_dim, lambd, W, K_arms, arms_index_array, initializeBanditParams='zeros'):
        # arms_index_array is multidimensional array of shape (K_arms, 1+arm dimensions), where 1 is for arm_index
        # initializeBanditParams should be 'random' for random bandit parameters initializations

        # Letting the length of adjacency matrix to be the userNum for use within the class
        # This actualLy stands for number of unique users, or number of clusters if applicable
        self.userNum = len(W)
        self.itemFeatureDim = itemFeatureDim
        self.latent_dim = latent_dim
        self.dim = self.itemFeatureDim + self.latent_dim
        self.K_arms = K_arms
        self.W = W

        # I don't need to keep track of A, only need A inverse, rank-one updates are all I need
        # will be updated with gathered information and wilL always be kept inverted (except for warm starting)
        self.AInv = (1 / lambd) * np.identity(self.dim * self.userNum)

        # wiLl be updated with gathered information
        self.b = np.zeros(self.dim * self.userNum)

        # to be learned, uppercase \hat{\theta} in the paper, Colin alg Line 6 has it in the UCB term, first term
        self.userTheta = np.zeros((self.dim, self.userNum))
        if initializeBanditParams == 'random':
            self.userTheta = l1NormalizationByCol(self.userTheta)

        # \bar{\Theta} in the paper, combination of user similarities with item-user preferences,
        # again Line 6, the term in the vec(.)
        self.barTheta = np.zeros((self.dim, self.userNum))

        # this is static, we just fix the kronecker multiplication of W and Identity matrix of proper size
        self.kronnedW = np.kron(np.transpose(W), np.identity(self.dim))

        # Confidence Bound - Optimistic part (CBO), second term of the UCB term on Line 9, 1st part of the variance term
        # not explicitly defined in the paper, but this holds for W.T * AInv * W in the first part of the variance term
        # we combine this with the content in getRec method below
        self.CBO = np.dot(np.dot(self.kronnedW, self.AInv), np.transpose(self.kronnedW))

        # Loop through all arms to store the information and initialize the arm objects
        self.arms = {}
        self.arms_index_array = arms_index_array
        for i in range(self.K_arms):
            self.arms[arms_index_array[i]] = FactorUCBArmStruct(arms_index_array[i], itemFeatureDim, latent_dim, lambd)

    def updateParameters(self, arm, click, userID):
        # the first element of the second term in the right hand-side of Line 8, do X and X transpose is the update on A
        X = vectorize(np.outer(arm.V, self.W.T[userID]))

        interim = Inv(self.AInv)
        interim.rank_one_update(X, X)
        self.AInv = interim.value

        # Line 9 in CoLin
        self.b += click * X

        self.userTheta = matrixize(np.dot(self.AInv, self.b), len(arm.V))  # how about normalizing this?

        # We use this in UCB calculations as the first term in UCB is arm context combined
        # with bandit parameters with other user effects plugged in
        self.barTheta = np.dot(self.userTheta, self.W)
        # If one is comparing bandit parameters across different algorithms, one should use this in that comparison
        # as the userTheta bandit parameter is the true bandit parameters for each user, if they were in a vacuum

        # update the confidence term in the UCB, again, if W matrix is identity, CBO is directly A inverse
        # this method gives the time it takes to calculate the CBO matrix,
        # that can be used for checking if this part is taking the Longest time or not
        # if this is not taking the Longest time, something is wrong either in this method on somewhere else
        # as this is the most costly step since we avoid inverting A
        stime = time.time()
        self.CBO = np.dot(np.dot(self.kronnedW, self.AInv), np.transpose(self.kronnedW))
        etime = time.time()

        # Update arm parameters too
        arm.updateParameters(self.barTheta, click, userID)
        return etime - stime

    def getRec(self, arm_info, userCluster, alpha=0.25, alpha2=0.25):
        # Instead of calculating the UCB for each arm and looping over that calculation, here we do it in one go.
        # Just get the rows of a column-vector to denote the UCB of each arm.
        # Here the assumption on the data is as follows:
        # Each row stores one item, the first column stores the item ID (not used within this method),
        # the other columns (there should be self.dim many of them) store the contextual information of that item.
        # This part calculates the UCB of each item and returns an array where the
        # UCB of i-th item in items info is in the i-th position of the returned array
        item_temp_features = np.empty([len(arm_info), self.dim * self.userNum])
        meanVector = np.empty(len(arm_info))
        varianceVector2 = np.empty(len(arm_info))
        arm = None
        for i in range(len(arm_info)):
            if arm_info[i, 0] in self.arms:
                arm = self.arms[arm_info[i, 0]]

            # if this is true, then v of that article is not populated with its original context, it should be
            if arm.checkv:
                arm.contextsIn(arm_info[i, 1:])

            TempFeatureM = np.zeros(shape=(self.dim, self.userNum))
            TempFeatureM.T[userCluster] = arm.V
            item_temp_features[i, :] = vectorize(TempFeatureM)
            meanVector[i] = np.dot(self.barTheta.T[userCluster], arm.V)
            varianceVector2[i] = np.sqrt(np.dot(np.dot(self.barTheta.T[userCluster][self.itemFeatureDim:], arm.CInv),
                                                self.barTheta.T[userCluster][self.itemFeatureDim:]))
        varianceMatrix = np.sqrt(np.dot(np.dot(item_temp_features, self.CBO), item_temp_features.T))
        UCBs = meanVector + alpha * np.diag(varianceMatrix) + alpha2 * varianceVector2

        # Sort the UCBs in decreasing order
        item_positions = np.argsort(UCBs)[::-1]

        # Reorder the rows in arm info to reflect the ordering of the UCBs
        items = []
        for i in range(len(arm_info)):
            items.append(arm_info[item_positions[i]])
        # returns the items with ID first and features next in a new order where the first item has the highest UCB
        # hence allows for multiple selections
        return items

    def return_userTheta_AInv_V_CInv(self):
        # returns the userTheta, AInv; V vectors and CInv matrices for each arm
        # these are enough to warm start another instance of the same user-relation set along with same set of arms
        return {'userThetas': self.userTheta.T, 'AInv': self.AInv,
                'V_set': [self.arms[self.arms_index_array[i]].V for i in range(self.K_arms)],
                'CInv_set': [self.arms[self.arms_index_array[i]].CInv for i in range(self.K_arms)]}
        # We return the userTheta transposed because that puts the users on the rows, easier to compare parameters
        # userTheta is read as transposed in Simulator, if it is not transposed here, it should be fixed in Simulator

    def fitParams(self, userTheta, A_inverse_in=False, AInv=np.empty(1),
                  latents_in=False, V=np.empty(1), CInv=np.empty(1)):
        # Assumes that the userTheta, AInv, V and CInv has the correct structure and dimensions,
        # e.g., V is a list of vectors and CInv is a list of matrices
        # By default, A inverse and latent information are not assumed to be plugged in
        # If only userTheta is provided, then only userTheta is plugged in
        # To plug in A inverse as well, the second argument should be True, the third argument should be proper AInv
        # To plug in latent information, fourth argument should be True, and the next should contain the proper inputs
        self.userTheta = userTheta
        if A_inverse_in:
            self.AInv = AInv
        if latents_in:
            for i in range(self.K_arms):
                self.arms[self.arms_index_array[i]].V = V[i]
                self.arms[self.arms_index_array[i]].CInv = CInv[i]

    def warmStart(self, AInv=False, Latents=False):
        # Can be called after properly provided userTheta (and AInv) is injected to the object
        # By default, this assumes only userTheta is injected and hence only barTheta is updated
        # If both user Theta and AInv are injected, then the only argument should be True
        self.barTheta = np.dot(self.userTheta, self.W)
        if AInv:
            self.CBO = np.dot(np.dot(self.kronnedW, self.AInv), np.transpose(self.kronnedW))
            self.b = np.dot(np.linalg.inv(self.AInv), vectorize(self.userTheta))
            # here b is using the regular inverse since we do not have access to A
        if Latents:
            for i in range(self.K_arms):
                self.arms[self.arms_index_array[i]].d = np.dot(np.linalg.inv(self.arms[self.arms_index_array[i]].CInv),
                                                               self.arms[self.arms_index_array[i]].V)
                # here d is using the regular inverse since we do not have access to C

    def readParams(self, warmStartPaths):
        path_userTheta = str(warmStartPaths['userTheta'])
        userTheta = np.transpose(pd.read_csv(path_userTheta, header=None, sep=",").to_numpy())
        # here userTheta is transposed because CoLinStruct returns the userTheta transposed

        if warmStartPaths['AInv_in']:
            path_AInv = str(warmStartPaths['AInv'])
            AInv = pd.read_csv(path_AInv, header=None, sep=",").to_numpy()

            if warmStartPaths['Latents_in']:  # if plugging in AInv as well as the latents
                path_V = str(warmStartPaths['V'])
                V_list = pd.read_csv(path_V, header=None, sep=",").to_numpy()
                path_CInv = str(warmStartPaths['CInv'])
                CInv_list = pd.read_csv(path_CInv, header=None, sep=",").to_numpy()

                self.fitParams(userTheta, True, AInv, True, V_list, CInv_list)
                self.warmStart(True, True)

            else:  # if plugging only AInv
                self.fitParams(userTheta, True, AInv)
                self.warmStart(True)
        else:
            if warmStartPaths['Latents_in']:  # if plugging in the latents but not AInv
                path_V = str(warmStartPaths['V'])
                V_list = pd.read_csv(path_V, header=None, sep=",").to_numpy()
                path_CInv = str(warmStartPaths['CInv'])
                CInv_list = pd.read_csv(path_CInv, header=None, sep=",").to_numpy()

                self.fitParams(userTheta, False, np.empty(1), True, V_list, CInv_list)
                self.warmStart(False, True)

            else:  # if plugging only userTheta
                self.fitParams(userTheta)
                self.warmStart()
