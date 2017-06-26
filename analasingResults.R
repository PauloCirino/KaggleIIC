DataFiles <- dir(path = './data/', pattern = '.RDA')

finalResultList <- list()
for(file in DataFiles){
    fileName <- paste('data/', file, sep = '')
    load(file = fileName)
    finalResultList <- append(finalResultList, resultList)
}

numIters <- ( length(finalResultList) / 3 ) - 1


allBestResults <- data.frame()
booster <- finalResultList[[1]]
lastResult <- finalResultList[[2]]
bestResult <- finalResultList[[3]]

auxAllBestResults <- as.matrix(unlist(bestResult))
auxAllBestResults <- data.frame(t(auxAllBestResults))
auxAllBestResults <- cbind(iter = 0, 
                           auxAllBestResults)
allBestResults <- auxAllBestResults
for(i in 1:numIters){
    booster <- finalResultList[[i*3 + 1]]
    lastResult <- finalResultList[[i*3 + 2]]
    bestResult <- finalResultList[[i*3 + 3]]
    
    auxAllBestResults <- as.matrix(unlist(bestResult))
    auxAllBestResults <- data.frame(t(auxAllBestResults))
    auxAllBestResults <- cbind(iter = i, 
                               auxAllBestResults)
    
    
    allBestResults[i, 1:ncol(allBestResults)] <- auxAllBestResults
}


bestIter <- which.min(allBestResults$bestTestRMSEMean)
booster <- finalResultList[[bestIter*3 + 1]]
lastResult <- finalResultList[[bestIter*3 + 2]]
bestResult <- finalResultList[[bestIter*3 + 3]]

require('xgboost')
source('xgboostPreparation.R')

params <- list(booster = booster$booster,
               objective = booster$objective,
               silent = booster$silent, 
               nthread = 20,
               eta = booster$eta,
               gamma= booster$gamma,
               max_depth = booster$max_depth,
               min_child_weight = booster$min_child_weight,
               subsample = booster$subsample,
               colsample_bytree = booster$colsample_bytree,
               print_every_n = booster$print_every_n,
               max_delta_step = booster$max_delta_step )

model <- xgb.train( params = params,
                    data = trainData,
                    nrounds = 7500,
                    metrics = "RMSE",
                    maximize = FALSE)

results <- predict(model, testData)

SampleSubmission <- readr::read_csv(file = './data/SampleSubmission.csv')
SampleSubmission$Target <- results

write.csv(x = SampleSubmission, 'results.csv', row.names = FALSE)


