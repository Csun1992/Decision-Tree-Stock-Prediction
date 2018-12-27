from classifier import Classifier
from sklearn import tree

class DecisionTreeStockPrediction(Classifier):
    def __init__(self, microDataLoc, clusterNum=1, macroDataLoc="data/clusterData.txt"):
        Classifier.__init__(self, microDataLoc, clusterNum, macroDataLoc)

    def train(self):
        train, test, trainLabel, testLabel = self.trainTestSplit()
        clf = [tree.DecisionTreeClassifier() for i in range(self.clusterNum)]
        for i in range(self.clusterNum):
            clf[i].fit(train[i], trainLabel[i])
        return (clf, test, testLabel) # return test and testLabel to self.test() so no need to
                                      # recompute the testing data again



class DecisionTreeStockPredictionNoClustering(DecisionTreeStockPrediction):
    def __init__(self, microDataLoc):
        DecisionTreeStockPrediction.__init__(self, microDataLoc)

    def train(self):
        train, test, trainLabel, testLabel = self.trainTestSplit()
        clf = [tree.DecisionTreeClassifier() for i in range(self.clusterNum)]
        for i in range(self.clusterNum):
            clf[i].fit(train[i], trainLabel[i])
        return (clf, test, testLabel) # return test and testLabel to self.test() so no need to
                                      # recompute the testing data again

if __name__ == "__main__":

    # without clustering
    apple = DecisionTreeStockPrediction("data/appleTrainData.txt")
    apple.reportResult()

    # for the case when cluster = 3
    apple = DecisionTreeStockPrediction("data/appleTrainData.txt", clusterNum=3)
    apple.reportResult()
        
    # Case when 2 clusters
    apple = DecisionTreeStockPrediction("data/appleTrainData.txt", clusterNum=2)
    apple.reportResult()
       
    # Case when 4 clusters
    apple = DecisionTreeStockPrediction("data/appleTrainData.txt", clusterNum=4)
    apple.reportResult()

    # Case when 1 cluster
    apple = DecisionTreeStockPrediction("data/appleTrainData.txt", clusterNum=1)
    apple.reportResult()





    # without clustering
    apple = DecisionTreeStockPredictionNoClustering("data/appleTrainData.txt")
    apple.reportResult()

