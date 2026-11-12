# 🚗 Used Car Price Prediction

## Project Overview

Used Car Price Prediction is a Machine Learning application that estimates the resale price of used cars based on their specifications.

Users can enter car details through an interactive Streamlit interface and receive an estimated selling price in Indian Rupees (₹).

## Objectives

- Analyse used car pricing data.
- Perform data cleaning and preprocessing.
- Explore patterns using data visualization.
- Train and evaluate Machine Learning models.
- Build an interactive price prediction application.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit
- Joblib

## Dataset

**Source:** [CarDekho Vehicle Dataset – Kaggle](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)

The dataset contains:
- Car name
- Manufacturing year
- Selling price
- Kilometres driven
- Fuel type
- Seller type
- Transmission
- Owner type

## Project Workflow

1. Data collection and inspection.
2. Data cleaning and duplicate removal.
3. Exploratory Data Analysis (EDA).
4. Feature engineering and car age calculation.
5. Categorical encoding using One-Hot Encoding.
6. Train-test splitting.
7. Model training and comparison.
8. Model evaluation using MAE, RMSE and R².
9. Interactive application development using Streamlit.

## Machine Learning Models

- Linear Regression
- Random Forest Regressor

The models are compared using evaluation metrics to select the best-performing model.

## Application Features

- Interactive car selection.
- Kilometres driven input.
- Fuel type and transmission selection.
- Seller and owner details.
- Car age input.
- Estimated resale price in Indian Rupees.

## Project Structure

```text
Used_Car_Price_Prediction/
│
├── data/
├── app.py
├── data_inspection.py
├── eda.py
├── preprocessing.py
├── used_car_price_model.pkl
├── .gitignore
└── README.md
```

## Installation and Execution

**1. Clone the repository**

```bash
git clone https://github.com/reenajacintha/Used_Car_Price_Prediction.git
```

**2. Open the project folder**

```bash
cd Used_Car_Price_Prediction
```

**3. Install dependencies**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib
```

**4. Run the application**

```bash
streamlit run app.py
```

## Future Enhancements

- Improve prediction accuracy.
- Add more car brands and models.
- Include additional vehicle specifications.
- Deploy the application online.

## Author

**Reena Jacintha**

B.Tech – Artificial Intelligence and Machine Learning

GitHub: [reenajacintha](https://github.com/reenajacintha)

---

⭐ Thank you for visiting this project!
