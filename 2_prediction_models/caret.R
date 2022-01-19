# Title:        Prediction models for TDA review paper
# Author:       Ewan Carr
# Started:      2020-02-26

library(tidyverse)
library(here)
library(caret)
library(glmnet)
library(here)
library(tidyverse)
library(fs)
library(furrr)
plan(multicore)
set.seed(42)

extract_auc <- function(m) {
  m$results[m$results[,1] == m$bestTune[1,1] &
            m$results[,2] == m$bestTune[1,2], 3]
}

extract_vimp <- function(m) {
  vi <- varImp(m, scale = FALSE)
  vi_nonull <- vi$importance[vi$importance$Overall != 0, ]
  return(arrange(tibble(predictors(m), vi_nonull), -vi_nonull))
}

# Load X/Y --------------------------------------------------------------------

setwd(here("2_prediction_models"))
filenames <- list.files("processed", pattern = "*.csv")
csv <- map(filenames, ~ read_csv(paste0("processed/", .x)))
names(csv) <- str_replace(filenames, "\\.csv$", "")

# Clean both outcomes ---------------------------------------------------------

# Binary remission
ybin <- csv %>%
    pluck("Ybin") %>%
    mutate(y = case_when(hdremit.all == 0 ~ "NO",
                         hdremit.all == 1 ~ "YES"),
           y = factor(y, levels = c("NO", "YES"))) %>%
    pluck("y")

ycon <- csv %>% pluck("Ycon") %>%
    pluck("mdpercadj")

# Select a single set of landscape variables ----------------------------------

# We're using "L10_50"
landscapes <- csv$L10_50 %>% select(-id)

# Select a single dimension only ----------------------------------------------

# For the review paper, we only want D1 or D2.
landscapes_D1 <- landscapes %>% select(starts_with("D1"))
landscapes_D2 <- landscapes %>% select(starts_with("D2"))

# Select baseline variables ---------------------------------------------------

baseline <- csv$X %>% as.data.frame()

###############################################################################
####                                                                      #####
####                    DEFINE FUNCTIONS TO FIT MODELS                    #####
####                                                                      #####
###############################################################################

remove_zero <- function(input) {
    nonzero_var <- input %>%
        summarise_all(var) %>%
        gather(measure, variance) %>%
        filter(variance > 0) %>%
        pluck("measure")
    return(input[nonzero_var])
}

get_best_result = function(caret_fit) {
  best = which(rownames(caret_fit$results) == rownames(caret_fit$bestTune))
  best_result = caret_fit$results[best, ]
  rownames(best_result) = NULL
  best_result
}

# GLMNET, continuous outcome --------------------------------------------------
fit_continuous <- function(X, y, reps = 100) {
    ctrl_continuous = trainControl(method = "repeatedcv",
                                   number = 3,
                                   repeats = reps)
    M <- train(X, y,
          method = "glmnet",
          trControl = ctrl_continuous,
          preProcess = c("center", "scale"),
          metric = "Rsquared")
    return(M)
}

# GLMNET, binary outcome ------------------------------------------------------
fit_binary <- function(x, y, reps = 100) {
    ctrl_binary  <- trainControl(method          = "repeatedcv",
                                 number          = 3,
                                 repeats         = reps,
                                 classProbs      = TRUE,
                                 summaryFunction = twoClassSummary,
                                 savePredictions = TRUE)
    M <- train(x,
               y,
               method     = "glmnet",
               metric     = "ROC",
               preProcess = c("center", "scale"),
               trControl  = ctrl_binary)
    return(M)
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

# Construct input combinations ------------------------------------------------
outcomes = list(ybin = ybin,
                ycon = ycon)
features = list(d1 = landscapes_D1,
                d2 = landscapes_D2,
                db = bind_cols(landscapes_D1, landscapes_D2),
                bl = baseline,
                b1 = bind_cols(landscapes_D1, baseline),
                b2 = bind_cols(landscapes_D2, baseline),
                bb = bind_cols(landscapes_D1, landscapes_D2, baseline))
all_inputs <- cross2(features, outcomes)

# Label models
model_names <- cross2(names(features), names(outcomes)) %>%
    map_chr(~ paste0(.x[[1]], "_", .x[[2]]))
names(all_inputs) <- model_names

# Remove features with zero variance ------------------------------------------
non_zero <- future_map(all_inputs, function(x) {
                           x[[1]] <- remove_zero(x[[1]])
                           return(x) }) %>%
    discard(~ ncol(.x[[1]]) == 0)

# Run models ------------------------------------------------------------------
xy <- non_zero$bb_ybin
f1 <-  model_picker(xy[[1]], xy[[2]], reps = 10)
extract_auc(f1)

all_models <- future_map(non_zero,
                         ~ model_picker(.x[[1]], .x[[2]], reps = 100))

save(all_models, file = paste0("saved_outputs/",
                                format(Sys.time(),
                                       "%Y_%m_%d_%H%d"), ".Rdata"))
