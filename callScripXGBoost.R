source('xgboostPreparation.R')

TS <- as.numeric(Sys.time())
set.seed(TS)

SAVE_EACH_MINS <- 30

booster <- 'gbtree'
nrounds <- 10000
print_every_n <- 100
early_stopping_rounds <- 500
nfold <- 30
objective <- "reg:linear"
metrics <- "rmse"
silent <- 0
nthread <- 6
eta <- c(0.1, 0.05, 0.01)
min_child_weight <- c(2, 5, 10)
max_depth <- c(2, 4, 6)
gamma <- c(0, 2)
max_delta_step <- c(0) 
subsample <- c(0.5, 0.75, 1)
colsample_bytree <- c(0.5, 0.75, 1)
colsample_bylevel <- c(0.5, 0.75, 1)

paramTable <- expand.grid(booster = booster,
                          nrounds = nrounds,
                          print_every_n = print_every_n,
                          early_stopping_rounds = early_stopping_rounds,
                          nfold = nfold,
                          objective = objective,
                          metrics = metrics,
                          silent = silent, 
                          nthread = nthread, 
                          eta = eta, 
                          min_child_weight = min_child_weight, 
                          max_depth = max_depth,
                          gamma = gamma,
                          max_delta_step = max_delta_step, 
                          subsample = subsample, 
                          colsample_bytree = colsample_bytree,
                          colsample_bylevel = colsample_bylevel)

callingOrder <- sample(nrow(paramTable))


resultList <- list()
lastSavedTime <- 0
for (paramIndex in callingOrder){
    iterParams <- paramTable[paramIndex, ]
    params <- list(booster = iterParams$booster,
                   objective = iterParams$objective,
                   silent = iterParams$silent, 
                   nthread = iterParams$nthread,
                   eta = iterParams$eta,
                   gamma= iterParams$gamma,
                   max_depth = iterParams$max_depth,
                   min_child_weight = iterParams$min_child_weight,
                   subsample = iterParams$subsample,
                   colsample_bytree = iterParams$colsample_bytree,
                   print_every_n = iterParams$print_every_n,
                   max_delta_step = iterParams$max_delta_step,
                   colsample_bylevel = colsample_bylevel )
    
    cv <- xgb.cv(params = params,
                 data = trainData,
                 nrounds = iterParams$nrounds,
                 nfold = iterParams$nfold,
                 metrics = iterParams$metrics,
                 early_stopping_rounds = iterParams$early_stopping_rounds,
                 maximize = FALSE)
    
    lastIter <- cv$niter
    endTrainRMSEMean <-cv$evaluation_log$train_rmse_mean[lastIter]
    endTrainRMSEStd <- cv$evaluation_log$train_rmse_std[lastIter]
    endTestRMSEMean <-cv$evaluation_log$test_rmse_mean[lastIter]
    endTestRMSEStd <- cv$evaluation_log$test_rmse_std[lastIter]
    
    bestIter <- cv$best_iteration
    bestNTrees <- cv$best_ntreelimit
    bestTrainRMSEMean <-cv$evaluation_log$train_rmse_mean[bestIter]
    bestTrainRMSEStd <- cv$evaluation_log$train_rmse_std[bestIter]
    bestTestRMSEMean <-cv$evaluation_log$test_rmse_mean[bestIter]
    bestTestRMSEStd <- cv$evaluation_log$test_rmse_std[bestIter]

    result <- list(params = params,
                   lastIterList = list( lastIter = lastIter,
                                        endTrainRMSEMean = endTrainRMSEMean,
                                        endTrainRMSEStd = endTrainRMSEStd,
                                        endTestRMSEMean = endTestRMSEMean,
                                        endTestRMSEStd = endTestRMSEStd ),
                   
                   bestIterList = list( bestIter = bestIter,
                                        bestNTrees = bestNTrees,
                                        bestTrainRMSEMean = bestTrainRMSEMean,
                                        bestTrainRMSEStd = bestTrainRMSEStd,
                                        bestTestRMSEMean = bestTestRMSEMean,
                                        bestTestRMSEStd = bestTestRMSEStd)
                   )
    
    resultList <- append(resultList, result)
    
    if ( (as.numeric(Sys.time()) - lastSavedTime) > SAVE_EACH_MINS ){
        save(resultList, file = paste('data/resultList_', TS, '.RDA', sep = ''))
    }
}