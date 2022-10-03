from CoLinStruct import *
from FactorUCBStruct import *
from CoLinFactorUCB_utils import *
from preprocessMLens import preprocessMovieLens1M


def batchUpdatesInit():
    moviesRecorded = []
    clicksRecorded = []
    clustersRecorded = []
    appended = 0
    return moviesRecorded, clicksRecorded, clustersRecorded, appended


def runMovieLens(colinfactor, path_input, path_output, warmStartPaths, iFD, lFD, lambd, initBanditParams, alpha, alpha2,
                 dim_contexts=18, top_users=1000, numClust=100, sparsity=100, top_movies=50, embeddingSize=50,
                 stopAfter=15000, recordEvery=1000, skip=0, batchUpdates=1):

    # preprocess the data, run CF on the desired subset, return that subset of information
    data, movie_features, users, W = preprocessMovieLens1M(path_input, sparsity, dim_movie_contexts=dim_contexts,
                                                           top_users=top_users, numClust=numClust,
                                                           top_movies=top_movies, embeddingSize=embeddingSize)
    print("Performed the proper preprocessing of the dataset.")
    moviesRecorded, clicksRecorded, clustersRecorded, appended = batchUpdatesInit()

    algObject = None
    if colinfactor == 'colin':
        algObject = ColinConstruct(iFD, lambd, alpha, W, initBanditParams)
    elif colinfactor == 'factor':
        algObject = FactorUCBUserStruct(itemFeatureDim=iFD, latent_dim=lFD, lambd=lambd, W=W,
                                        K_arms=len(movie_features),
                                        arms_index_array=list(movie_features['movie_id']),
                                        initializeBanditParams=initBanditParams)
    else:
        print("Problem in the algorithm selection phase. Exiting...")
        exit()

    aligned_time_steps = 0  # keeps track of the data points where the recommendation matched the actual data
    time_steps = 0  # keeps track of the overall number of data points considered, excludes warm start points
    lines = 0  # keeps track of all the data points that was 'seen', includes warm start points:
    # those are the points that aren't explicitly used in the current simulation

    cumulative_rewards = 0
    cumulative_rewards_list = []
    record_aligned_time_steps = 0  # aligned time steps in the previous iteration
    record_cumulative_rewards = 0  # cumulative rewards in the previous iteration
    CTR = []
    output = colinfactor + "_object"

    if skip > 0:
        algObject.readParams(warmStartPaths)
        print("Parameters are successfully read.")

    starttime = time.time()
    print("Starting")
    for i in range(len(data)):
        lines += 1

        if lines > skip:
            time_steps += 1
            if max(time_steps, lines) > stopAfter:
                break

            user_id = data.loc[i, "user_id"]
            # if clustering, user id will stand for the cluster that this particular user belongs, clusters start from 0
            clusterAssgn = np.asarray(users.query("user_id == @user_id")["cluster"])[0]

            movie_id = data.loc[i, "movie_id"]
            reward = data.loc[i, "reward"]  # Obtain rewards

            if i % 2500 == 0:
                print(" \nSTEP " + str(i) + ", AND TIME FROM START" + str(time.time() - starttime) + " =" + "=" * 15)

            # Find policy's chosen arm based on input covariates at current time step
            arm_recs = algObject.getRec(movie_features.to_numpy(), clusterAssgn, alpha, alpha2)
            chosen_arm = arm_recs[0]
            # Check if arm index is the same as data arm (i.e., same actions were chosen)
            if movie_id == chosen_arm[0]:
                movie = None  # this arm selection part should be made homogeneous across two methods
                if colinfactor == 'colin':
                    movie = chosen_arm[1:]
                elif colinfactor == 'factor':
                    movie = algObject.arms[movie_id]

                if batchUpdates > 1:
                    moviesRecorded.append(movie)
                    clicksRecorded.append(reward)
                    clustersRecorded.append(clusterAssgn)
                    appended += 1
                    aligned_time_steps += 1
                    cumulative_rewards += reward
                    cumulative_rewards_list.append(cumulative_rewards)

                    if appended > 0 & appended % batchUpdates == 0:
                        # do the updates
                        for ij in range(batchUpdates):
                            algObject.updateParameters(moviesRecorded[ij],
                                                       clicksRecorded[ij], clustersRecorded[ij])
                        # initialize the recorders
                        moviesRecorded, clicksRecorded, clustersRecorded, appended = batchUpdatesInit()
                else:
                    algObject.updateParameters(movie, reward, clusterAssgn)
                    aligned_time_steps += 1
                    cumulative_rewards += reward
                    cumulative_rewards_list.append(cumulative_rewards)

            if time_steps % recordEvery == 0:
                current_CTR = (cumulative_rewards - record_cumulative_rewards) / \
                              max(aligned_time_steps - record_aligned_time_steps, 1)
                CTR.append(current_CTR)
                record_cumulative_rewards = cumulative_rewards
                record_aligned_time_steps = aligned_time_steps

                if time_steps & (recordEvery * 10) == 0:
                    print(str(colinfactor) + " cumulative reward " + str(cumulative_rewards) + ", aligned step " +
                          str(aligned_time_steps) + "at simulation time " + str(time_steps) + " and " +
                          str(time.time() - starttime) + " seconds from start.")
                    print("Total running CTR lift is " + str(cumulative_rewards / max(aligned_time_steps, 1)))

                if time_steps % (recordEvery * 10) == 0:
                    res = printResults(algObject, path_output, CTR, str(colinfactor))
                    if not res:
                        print("Exiting...")
                        break

    printResults(algObject, path_output, CTR, str(colinfactor))  # print results just one more time before exiting

    return {'aligned_time_steps': aligned_time_steps, 'cumulative_rewards': cumulative_rewards,
            'cumulative_rewards_list': cumulative_rewards_list, 'CTR': CTR, str(output): algObject}
