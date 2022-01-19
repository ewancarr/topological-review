# Title:        Prepare features (discritised landscapes and baseline clinical
#               variables) for prediction modelling
# Author:       Ewan Carr
# Date:         2021-02-03

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.impute import KNNImputer
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from tda.prediction import (make_landscape,
                            process_repeated_measures,
                            combine_landscapes)
from pathlib import Path

proc =  Path('processed')

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                         Load data; select sample                          ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# Load baseline data, with outcome/features -----------------------------------
gendep = pd.read_csv("data/GENDEP/dataClin.csv")

# Select 232 from Raquel's 2016 paper -----------------------------------------
selected = pd.read_csv('data/GENDEP/randomised_escitalopram_232.csv')

chosen_ids = selected['subjectid']

# Recode required variables ---------------------------------------------------
required = list(gendep)
for i in ['Row.names', 'anxdep', 'melanch', 'hrsd.total', 'hamd16wk0',
          'bdi.total', 'madrs.total', 'children.rec', 'atyp', 'occup.rec',
          'mscore', 'marital.rec', 'atscore']:
    required.remove(i)

# Select required columns -----------------------------------------------------
features = selected[required]

# Add missing measures --------------------------------------------------------

# Assumption: bdi0 = bdi.total
# Assumption: melanch.x = melanch
# Assumption: hamw0.re = hrsd.total
# Assumption: anxdep.y = anxdep
# Assumption: madrs.total = madrs0
# Assumption: mscore = mscore.y

features = pd.concat([features,
                      selected[['bdi0', 'melanch.x', 'hamw0.re', 'anxdep.y',
                                'madrs0', 'mscore.y']]], axis=1)

# Create dummy variables ------------------------------------------------------
dummies = pd.get_dummies(selected[['children', 'occup', 'marital']])
features = pd.concat([features, dummies], axis=1)

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃               PREPARE NON-LANDSCAPE VARIABLES FOR MODELLING               ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# FEATURES ====================================================================

# Drop features that shouldn't be included in modelling
selected_features = features.drop(columns=['drug', 'hdremit.all', 'subjectid',
                                           'mdpercadj', 'bloodsampleid.x'])


# Impute missing values -------------------------------------------------------
# imputer = KNNImputer(n_neighbors=2)
lab = list(selected_features)
selected_features_imputed = pd.DataFrame(imputer.fit_transform(selected_features),
        columns=lab)

# Save to CSV
selected_features_imputed.to_csv(proc / 'X.csv', index=False)

# OUTCOMES ====================================================================
features['hdremit.all'].to_csv(proc / 'Ybin.csv', header=True)
features['mdpercadj'].to_csv(proc / 'Ycon.csv', header=True)

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                      COMPUTE DISCRETISED LANDSCAPES                       ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# NOTE: we have two sources of repeated measures data. For now, I'm
#       combining both.

# ========================= NEW DATA (January 2021) ========================= #

repeat = pd.read_stata(Path('data/GENDEP/gendep-share-repeated-measures.dta')

# Select sample (n=232, to match 2016 paper) ==================================
repeat = repeat[repeat['subjectid'].isin(chosen_ids)]

# Select weeks 0-4 ============================================================
repeat = repeat[repeat.week < 5]

# Select required variables ===================================================
req = ['subjectid', 'week', 'madrs1wk', 'madrs2wk', 'madrs3wk', 'madrs4wk',
       'madrs5wk', 'madrs6wk', 'madrs7wk', 'madrs8wk', 'madrs9wk', 'madrs10wk',
       'madrstotwk', 'hamd1wk', 'hamd2wk', 'hamd3wk', 'hamd4wk', 'hamd5wk',
       'hamd6wk', 'hamd7wk', 'hamd8wk', 'hamd9wk', 'hamd10wk', 'hamd11wk',
       'hamd12wk', 'hamd13wk', 'hamd14wk', 'hamd15wk', 'hamd17wk', 'hamd16wk',
       'hamdtotwk', 'bdi1wk', 'bdi2wk', 'bdi3wk', 'bdi4wk', 'bdi5wk', 'bdi6wk',
       'bdi7wk', 'bdi8wk', 'bdi9wk', 'bdi10wk', 'bdi11wk', 'bdi12wk',
       'bdi13wk', 'bdi14wk', 'bdi15wk', 'bdi16wk', 'bdi17wk', 'bdi18wk',
       'bdi19wk', 'bdi20wk', 'bdi21wk', 'bditotwk', 'gafwk']

repeat_2021 = repeat[req]

# ============ PREVIOUS VERSION OF REPEATED MEASURES DATA (2020) ============ #

repeat = pd.read_stata('data/GENDEP/gendep_core.dta')

# Select sample (n=232, to match 2016 paper) ==================================
repeat = repeat[repeat['subjectid'].isin(chosen_ids)]

# Select different sets of variables ==========================================
incl = 'subjectid|f[0-9][1-6]*score[0-4]$|madrs[0-4]$|hdrs[0-4]$|bdi[0-4]$'
repeat_2019 = repeat.filter(regex=incl)

# ============ MERGE THE TWO SETS OF REPEATED MEASURES DATA ================= #

# We have three options:
# 1. Use OLD _and_ NEW repeated measures data.
# 2. Use OLD only.
# 3. Use NEW only.

# Reformat 2021 repeated measures data ========================================
split = dict(tuple(repeat_2021.groupby('subjectid')))
processed_2021 = {}
for k, v in tqdm(split.items()):
    # Remove ID/week
    v.drop(labels=['subjectid', 'week'], axis=1, inplace=True)
    # If missing all repeated measures, replace with zeros to allow loop to
    # continue
    if v.isnull().all().all():
        v.fillna(0, inplace=True)
    # Scale
    sc = StandardScaler()
    processed_2021[k] = pd.DataFrame(sc.fit_transform(v))

# Repeat 2019 repeated measures data ==========================================
split = dict(tuple(repeat_2019.groupby('subjectid')))
processed_2019 = {k: process_repeated_measures(v) for k, v in split.items()}

# Merge the two datasets ======================================================
processed = {}
for k in tqdm(processed_2019.keys()):
    m = pd.concat([processed_2019[k],
                   processed_2021[k]], axis=1)
    i = KNNImputer()
    processed[k] = pd.DataFrame(i.fit_transform(m))

# Choose which dataset to use
processed = processed_2019

for k, v in processed.items():
    ,

# Create persistence ==========================================================
landscapes = []
for m in [10]:
    # MAS
    for r in [50]:
        # x_max
        L = {k: make_landscape(v, m, r) for k, v in processed.items()}
        L = combine_landscapes(L)
        landscapes.append([L, m])
        L.to_csv(proc / ('L' + str(m) + '_' + str(r) + '.csv'))

# Save repeated measures for use in prediction modelling ======================
pd.merge(repeat_2019, repeat_2021, on='subjectid'). \
    drop(labels=['week'], axis=1). \
    set_index('subjectid'). \
    to_csv(proc / 'RM.csv')
