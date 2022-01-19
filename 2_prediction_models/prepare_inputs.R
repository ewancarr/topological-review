# Title:        Prediction models for TDA review paper
# Author:       Ewan Carr
# Started:      2020-02-26, updated 2021-02-03

library(tidyverse)
library(here)
library(caret)
library(glmnet)
library(here)
library(tidyverse)
library(furrr)
plan(multicore)
library(doParallel)
library(fs)
set.seed(42)

# Load X/Y --------------------------------------------------------------------

setwd(here("2_prediction_models"))
filenames <- list.files("processed", pattern = "*.csv")
csv <- future_map(filenames, ~ read_csv(paste0("processed/", .x)),
                  .options = furrr_options(seed = 42))
names(csv) <- str_replace(filenames, "\\.csv$", "")

# Clean both outcomes ---------------------------------------------------------

# Binary remission
ybin <- csv %>%
    pluck("Ybin") %>%
    mutate(y = case_when(hdremit.all == 0 ~ "NO",
                         hdremit.all == 1 ~ "YES"),
           y = factor(y, levels = c("NO", "YES"))) %>%
    pluck("y")

# Percentage improvement
ycon <- csv %>%
    pluck("Ycon") %>%
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

# Construct input combinations ------------------------------------------------
results <- list()
outcomes = list(ybin = ybin,
                ycon = ycon)
features = list(d1 = landscapes_D1,
                d2 = landscapes_D2,
                db = bind_cols(landscapes_D1, landscapes_D2),
                bl = baseline,
                b1 = bind_cols(landscapes_D1, baseline),
                b2 = bind_cols(landscapes_D2, baseline),
                bb = bind_cols(landscapes_D1, landscapes_D2, baseline))
all_models <- cross2(features, outcomes)

# Split and save inputs -------------------------------------------------------

model_names <- cross2(names(features), names(outcomes)) %>%
    map_chr(~ paste0(.x[[1]], "_", .x[[2]]))

walk2(all_models, model_names,
      ~ save(.x,
             file = paste0("inputs/", .y, ".Rdata")))

# Save index ------------------------------------------------------------------

index <- data.frame(model_names) %>%
    mutate(id = 1:nrow(.))
save(index, file = "inputs/index.Rdata")
