This repository contains code for synthetic data modelling related to the SURVAYA population.

This project focuses on synthetic data generation and modelling for the AYA framework. It provides tools and utilities for tuning models (eg. hyperparameter tuning; grid search) and evaluating synthetic data across runs (ie. tally distributions, feature importance scores, odds ratio calculations).

The code in this repository was implemented to model logistic regression, random forests, and multilayer percptron classifier to investigate the stability of synthetic data (SD) on which the models were trained in comparison to real data (RD) on which the models were tested. Models were initialized using Scikit-Learn. A more extensive description of the methodology that this repository represents can be found in the associated scientific publication: 

Where have the contents of this repository been used and reported?
The stability of synthetic data was investigated in a rare and heterogeneous healthcare demographic: adolescents and young adults with cancer. The findings of this investigation can be found in the associated scientific publication:

How was this work funded?
This work and the associated scientific publication were predominantly supported by the European Union’s Horizon 2020 research and innovation programme through The STRONG-AYA Initiative (Grant agreement ID: 101057482).

What are the main libraries that this research code relied on?
The synthetic data was generated using Synthetic Data Vault (SDV) (https://github.com/sdv-dev/SDV), and Differentially Private - Conditional Generative Adversarial Networks ( DP-CGAN) (https://github.com/sunchang0124/dp_cgans). The evaluations were performed using sklearn (https://github.com/scikit-learn/scikit-learn).
