# Title:        Prediction models for TDA review paper
# Author:       Ewan Carr
# Started:      2020-02-26

"
For the review paper, we'll either use binary remission (hdremit.all) or
percentage improvement (mdpercadj). So this script fits models for both
outcomes.
"

library(caret)
library(glmnet)
set.seed(42)

###############################################################################
####                                                                      #####
####                    DEFINE FUNCTIONS TO FIT MODELS                    #####
####                                                                      #####
###############################################################################

extract_auc <- function(m) {
  m$results[m$results[,1] == m$bestTune[1, 1] &
            m$results[,2] == m$bestTune[1, 2], 3]
}

extract_vimp <- function(m) {
  vi <- varImp(m, scale = FALSE)
  vi_nonull <- vi$importance[vi$importance$Overall != 0, ]
  return(arrange(tibble(predictors(m), vi_nonull), -vi_nonull))
}

remove_zero <- function(input) {
    nonzero <- nearZeroVar(input, saveMetrics = TRUE)
    nonzero$feat <- row.names(nonzero)
    keepers <- nonzero$feat[!nonzero$nzv]
    return(input[keepers])
}

get_best_result = function(caret_fit) {
  best = which(rownames(caret_fit$results) == rownames(caret_fit$bestTune))
  best_result = caret_fit$results[best, ]
  rownames(best_result) = NULL
  best_result
}

# GLMNET, continuous outcome --------------------------------------------------

fit_continuous <- function(X, y, reps = 100) {
    library(doParallel)
    cl <- makePSOCKcluster(24)
    registerDoParallel(cl)
    ctrl_continuous = trainControl(method = "repeatedcv",
                                   number = 3,
                                   repeats = reps)
    if ("subjectid" %in% names(X)) {
        X <- X[ , !(names(X) == "subjectid")]
    }
    dat <- cbind(remove_zero(X), y)
    M <- train(y ~ .,
               data = dat,
          method = "glmnet",
          trControl = ctrl_continuous,
          preProcess = c("center", "scale"),
          metric = "Rsquared")
    return(M)
    stopCluster(cl)
}

# GLMNET, binary outcome ------------------------------------------------------

fit_binary <- function(X, y, reps = 100) {
    library(doParallel)
    cl <- makePSOCKcluster(24)
    registerDoParallel(cl)
    ctrl_binary  <- trainControl(method          = "repeatedcv",
                                 number          = 10,
                                 repeats         = reps,
                                 classProbs      = TRUE,
                                 summaryFunction = twoClassSummary,
                                 savePredictions = TRUE)
    if ("subjectid" %in% names(X)) {
        X <- X[ , !(names(X) == "subjectid")]
    }
    dat <- cbind(remove_zero(X), y)
    M <- train(y ~ .,
               data = dat,
               method     = "glmnet",
               metric     = "ROC",
               preProcess = c("center", "scale"),
               trControl  = ctrl_binary)
    return(M)
    stopCluster(cl)
}

model_picker <- function(X, y, reps = 100) {
    if (is.factor(y)) {
        fit_binary(X, y, reps)
    } else {
        fit_continuous(X, y, reps)
    }
}

###############################################################################
####                                                                      #####
####                              FIT MODELS                              #####
####                                                                      #####
###############################################################################

# Inputs should be:
# 1: Counter
# 2: Number of repeats
# 3: Number of folds
# 4: Input directory, absolute path
# 5: Output directory, absolute path

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2)
id <- as.numeric(args[1])
n_repeats <- as.numeric(args[2])

# Get inputs, for given counter -----------------------------------------------

load("inputs/index.Rdata")
model <- as.character(index$model_names[index$id == id])
load(paste0("inputs/", model, ".Rdata"),
     verbose = TRUE)
X = .x[[1]]
y = .x[[2]]
rm(.x)

# Run models ------------------------------------------------------------------

fit <- model_picker(X, y, reps = n_repeats)
save(fit, file = paste0("outputs/", model, ".Rdata"))
