# Sepsis Prediction Using AI: LSTM vs Random Forest

## Project Overview
Examining the impact of AI on healthcare, particularly in assisting with diagnosis, using the PhysioNet ICU Sepsis Dataset.

## Models Implemented
- **Traditional ML**: Random Forest
- **Deep Learning**: LSTM (Long Short-Term Memory)

## Dataset
- **Source**: PhysioNet/CinC Challenge 2019 
- **Patients**: 5,044 ICU stays
- **Features**: 40 clinical variables (vitals + labs)
- **Target**: Early prediction of sepsis onset

## Project Structure
- `data/`: Raw and processed datasets
- `src/`: All Python scripts
- `models/`: Saved trained models
- `results/`: Figures, metrics, and logs
- `report/`: Final coursework report

## Running the Project
1. Data exploration: `python src/01_data_exploration.py`
2. Preprocessing: `python src/04_preprocessing.py`
3. Train models: `python src/05_train_random_forest.py` and `python src/06_train_lstm.py`
4. Evaluate: `python src/07_evaluate_models.py`


Robert Chitic University Of Greenwich
Module: Artificial Intelligence Applications
