from runDatasets import *
import argparse
import yaml

if __name__ == '__main__':

    np.seterr(invalid='ignore')  # suppress all numpy warnings

    parser = argparse.ArgumentParser(description='Collaborative Bandits')
    parser.add_argument('--alg', dest='alg', help='Select a specific algorithm, could be colin or factor')
    parser.add_argument('--data', dest='data', help='Select a specific dataset, could only be movielens for now')
    args = parser.parse_args()

    algName = str(args.alg)  # should be colin or factor
    if not (algName == 'colin' or algName == 'factor'):
        print('Choose colin or factor as the alg. Exiting...')
        exit()

    datasetName = str(args.data)  # should be movielens
    if not datasetName == 'movielens':  # or datasetName == 'yahoo'
        print('Choose movielens as the dataset. Exiting...')
        exit()

    ymlFilename = 'ML_config.yaml'
    if datasetName == 'yahoo':
        ymlFilename = 'yahoo_config.yaml'

    with open(ymlFilename, 'r') as ymlfile:
        dataInfo = yaml.load(ymlfile, Loader=yaml.FullLoader)

    if datasetName == 'movielens':
        # for movielens
        # top_users: select top users with the most ratings
        # top_movies: select top most rated movies among top_users with the most ratings
        # embeddingSize: size of the embedding for the CF

        path_input = dataInfo['path_input']
        path_output = path_input + "/results"
        dim_contexts = dataInfo['itemFeatureDim']

        clusterNum = dataInfo['clusterNum'] if 'clusterNum' in dataInfo else 100
        top_users = dataInfo['top_users'] if 'top_users' in dataInfo else 1000000
        top_movies = dataInfo['top_movies'] if 'top_movies' in dataInfo else 50
        embeddingSize = dataInfo['embeddingSize'] if 'embeddingSize' in dataInfo else 50
        sparsity = min(dataInfo['sparsity'], clusterNum) if 'sparsity' in dataInfo else clusterNum

        stopAfter = dataInfo['stopAfter'] if 'stopAfter' in dataInfo else 10000
        recordEvery = dataInfo['recordEvery'] if 'recordEvery' in dataInfo else 100
        skip = dataInfo['skip'] if 'skip' in dataInfo else 0
        batchUpdates = dataInfo['batchUpdates'] if 'batchUpdates' in dataInfo else 1
        initBanditParams = dataInfo['initBanditParams'] if 'initBanditParams' in dataInfo else 'zeros'
        runCF = dataInfo['run_CF'] if 'run_CF' in dataInfo else True
        runKmeans = dataInfo['run_Kmeans'] if 'run_Kmeans' in dataInfo else True

        warmStartPaths = dataInfo['warmStart_input'] if 'warmStart_input' in dataInfo else ""
        iFD = dataInfo['itemFeatureDim'] if 'itemFeatureDim' in dataInfo else -1
        lFD = dataInfo['latent_dim'] if 'latent_dim' in dataInfo else -1

        lambd = dataInfo['lambda_'] if 'lambda_' in dataInfo else -1

        alpha = dataInfo['alpha'] if 'alpha' in dataInfo else 0.5 * (algName == 'colin') + 0.375 * (algName == 'factor')
        alpha2 = dataInfo['alpha2'] if 'alpha2' in dataInfo else -1

        runMovieLens(algName, path_input, path_output, warmStartPaths, iFD, lFD, lambd, initBanditParams, alpha, alpha2,
                     runCF, runKmeans, top_users=top_users, numClust=clusterNum, sparsity=sparsity,
                     top_movies=top_movies, embeddingSize=embeddingSize, stopAfter=stopAfter,
                     recordEvery=recordEvery, skip=skip, batchUpdates=batchUpdates)
