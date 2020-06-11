# DOTA-2-Dashboard
R Shiny Project to view DOTA 2 Analytics, including stats, association rules and neural networks for hero prediction to assist with drafting

Acknowledgements:  
DOTA 2 Data sourced from https://www.kaggle.com/devinanzelmo/dota-2-matches

Instructions to run:
install the required packages in R and Python

Download the following from the kaggle link above and place into the same folder as Dota_Dashboard.R
  chat.csv
  hero_names.csv
  match.csv
  players.csv

Dota_Dashboard.R contains the R Shiny application

To replicate any of the provided files:

The csv file "hero_drafting" is created from "Association Rules in DOTA 2 Hero Drafting/Hero_Drafting_Data_Preparation.ipynb"

The neural network states are created using the notebok"Hero_Prediction.ipynb", running this after setting the parameter "retrain" = True in the train functions for the FFNN and LTSM will retrain each model




