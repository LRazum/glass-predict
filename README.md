# **Phosphate Glass Virtual Laboratory**

**Project Status**: Early Prototype This project is currently in its initial development phase. It is a working version intended primarily for testing, research, and validation purposes, not a final production-ready tool.

A machine learning pipeline and interactive web app for predicting the properties of phosphate glasses based on their molar composition.

## **Predicted Properties**

* **Density** (g/cm³)  
* **Glass Transition Temperature** (Tg, °C)  
* **DC Conductivity** (30°C, S/cm)  
* **Activation Energy** (EDC, kJ/mol)

## **Key Features**

* **Physics-Informed Descriptors:** Automatically calculates Average Ion Radius, Charge, and Field Strength.  
* **XGBoost:** High-accuracy point predictions using regularized gradient boosting.  
* **Gaussian Process Regression (GPR):** Bayesian approach providing 95% Confidence Intervals (±).  
* **Streamlit App:** Real-time interactive testing interface.

## **Repository Structure**

glass-predict/  
├── app/                  \# Streamlit web app & trained models (.joblib)  
├── data/                 \# Raw and processed datasets  
├── models/               \# Training plots and artifacts  
├── scripts/              \# Preprocessing scripts  
├── gauss\_p\_regression\_train.py  
└── xgboost\_train.py

## **Installation**

Clone the repository and install dependencies:  
git clone \[https://github.com/LRazum/glass-predict.git\](https://github.com/LRazum/glass-predict.git)  
cd glass-predict  
pip install pandas numpy scikit-learn xgboost streamlit matplotlib joblib

## **Usage**

**1\. Launch the Virtual Lab (Web App)**  
Test new glass compositions in your browser:  
streamlit run app/app.py

**2\. Retrain Models (Optional)**  
If using a new dataset, run the training scripts:  
python xgboost\_train.py  
python gauss\_p\_regression\_train.py

## **Validation**

Models are strictly validated against overfitting (Small Data Paradox) using **5-Fold Cross-Validation** and **Y-Randomization** (Target Shuffling).  
*Developed for materials science research.*
