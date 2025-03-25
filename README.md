# QRT-Hackathon
My approach to the QRT hackathon at IIT Delhi on 2nd March 2025. The task was to analyse historical returns data and values of ~20 different unknown features' daily data, and use it to formulate an alpha, optimising for sharpe. I will not upload the source data or the problem statement due to confidentiality/copyright reasons, but the approaches here are general and can work across diverse situations.
## Overview
Basically we clean the data, remove entries that are unrecoverable, impute the rest of missing values, and then calculate the correlation of every feature with the returns and the standard deviation of returns.
This can be used to formulate alphas using general strategies. My alpha was `sum([feature*cor(returns)/(e**cor(stdev_returns)) for every feature available])` which aims to directly optimise for sharpe. It placed fifth among 190+ entrants.
## Permissions
This repo only contains code written by me, and you're free to reuse it.
