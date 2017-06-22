require('xgboost')
library('readr')

xgboostPreparation <- function(trainDataFileName, trainLabelsFileName, 
                               testDataFileName){
    trainData <- readr::read_csv(trainDataFileName)
    testData <- readr::read_csv(testDataFileName)
    
    trainLabels <- readr::read_csv(trainLabelsFileName)
    
    trainData <- xgboost::xgb.DMatrix(data = data.matrix(trainData),
                                      label = as.numeric( trainLabels$. ) )
    
    testData <- xgboost::xgb.DMatrix(data = data.matrix(testData))
    
    list(trainData = trainData,
         testData = testData)
}

auxList <- xgboostPreparation(  trainDataFileName = "./data/processedTrainData.csv",
                                trainLabelsFileName = "./data/TrainLabels.csv",
                                testDataFileName = "./data/processedTestData.csv"
                            )

trainData <- auxList$trainData
testData <- auxList$testData

rm( auxList )
rm( xgboostPreparation )
