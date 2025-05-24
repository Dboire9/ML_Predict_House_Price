# House Prices Prediction - Kaggle ML Course

This project is a simple pipeline for predicting house prices using the [Kaggle Home Data for ML Course](https://www.kaggle.com/competitions/home-data-for-ml-course) dataset. It uses Python, pandas, scikit-learn, and matplotlib.

## Features

- Loads and explores the training and test datasets
- Visualizes feature correlations and relationships
- Handles missing values and outliers
- Feature engineering (e.g., encoding categorical variables)
- Trains a linear regression model
- Evaluates model performance (R² and RMSE)
- Generates predictions for submission to Kaggle

## Usage

1. **Install dependencies**  
   Make sure you have Python 3 and the following packages:
   - pandas
   - numpy
   - matplotlib
   - scikit-learn

   You can install them with:
   ```
   pip install pandas numpy matplotlib scikit-learn
   ```

4. **Run the script**
   ```
   python3 training.py
   ```

5. **Output**
   - The script will print model metrics and create a `submission.csv` file for Kaggle submission.

## File Structure

```
home-data-for-ml-course/
├── data/
│   ├── train.csv
│   └── test.csv
├── training.py
└── submission.csv
```

## Notes

- The script uses log-transformed sale prices for training to reduce skewness.
- Only numeric features are used for modeling; categorical features are encoded as needed.
- Outliers in `GarageArea` are removed for better model performance.
- The final predictions are exponentiated to reverse the log transformation.

---

**Author:**  
Dorian Boiré

**License:**  
MIT
