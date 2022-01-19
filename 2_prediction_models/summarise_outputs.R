# Title:        Summarise prediction models for TDA review paper, run on Rosalind
# Author:       Ewan Carr
# Started:      2020-02-26

library(tidyverse)
library(here)
library(caret)
library(glmnet)
library(here)
library(tidyverse)
library(fs)
set.seed(42)

get_best_result = function(caret_fit) {
  best = which(rownames(caret_fit$results) == rownames(caret_fit$bestTune))
  best_result = caret_fit$results[best, ]
  rownames(best_result) = NULL
  return(best_result)
}

extract_auc <- function(m) {
  m$results[m$results[,1] == m$bestTune[1,1] &
            m$results[,2] == m$bestTune[1,2], 3]
}

extract_vimp <- function(m) {
  vi <- varImp(m, scale = FALSE)
  vi_nonull <- vi$importance[vi$importance$Overall != 0, ]
  return(arrange(tibble(predictors(m), vi_nonull), -vi_nonull))
}

# Load saved models, estimated on Rosalind ------------------------------------

if ( FALSE ) {
outputs <- dir_ls(here("2_prediction_models", "outputs")) %>%
    enframe() %>%
    mutate(name = path_ext_remove(path_file(name)),
           model = map(value, function(x) { load(x); return(fit) })) %>%
    select(name, model)
}

# OR, load local outputs ------------------------------------------------------

outputs <- dir_ls(here("2_prediction_models", "saved_outputs"))

all_outputs <- list()
for (i in outputs) {
    load(i, verbose = TRUE)
    all_outputs[[i]] <- all_models
}

names(all_outputs) <- names(all_outputs) %>%
    path_file() %>%
    path_ext_remove()


# Summarise model fit ---------------------------------------------------------


extract_summary <- function(outputs) {
    bind_cols(names(outputs),
              map_dfr(outputs, get_best_result)) %>%
      rename(model = `...1`) %>%
      filter(str_detect(model, "ybin$")) %>%
      select(model, ROC, Sens, Spec) %>%
      mutate(landscapes = case_when(str_detect(model, "^[db]1") ~ "1st dimension",
                                    str_detect(model, "^[db]2") ~ "2nd dimension",
                                    str_detect(model, "^[db]b") ~ "Both dimensions",
                                    TRUE ~ "No landscapes"),
             outcome = if_else(str_detect(model, "ybin"),
                                          "hdremit.all",
                                          "mdpercadj"),
             features = if_else(str_detect(model, "^b"),
                                "Baseline features", "No baseline features")) %>%
      return()
}


latest <- "2021_05_06_1206"
model_summary <- bind_cols(outputs,
                           map_dfr(outputs$model, get_best_result)) %>%
           perf =  coalesce(ROC, Rsquared) %>%
    select(landscapes, outcome, features, perf)

p <- model_summary %>%
    ggplot(aes(x = str_wrap(landscapes, 1),
               y = perf,
               colour = features,
               group = features)) +
    geom_text(aes(label = sprintf("%.3f", perf))) +
    geom_line(alpha = 0.5) +
    facet_wrap(~ outcome, scales = "free_y") +
    scale_colour_brewer(type = "qual", palette = 2) +
    theme_light() +
    theme(axis.title.x = element_blank())

ggsave(p, file = "~/summary.png",
       width = 8,
       height = 4,
       dpi = 300)

# Number of influential features ----------------------------------------------

a <- extract_vimp(outputs$model[outputs$name == "bl_ybin"][[1]])
b <- extract_vimp(outputs$model[outputs$name == "b1_ybin"][[1]])

nrow(a)
nrow(b)

b %>% filter(str_detect(`predictors(m)`, "^D")) %>% nrow()
b %>% filter(!str_detect(`predictors(m)`, "^D")) %>% nrow()