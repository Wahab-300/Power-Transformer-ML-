# Power Transformer - Concrete Strength Prediction

Transform skewed data into normal distributions for better ML model performance using Box-Cox and Yeo-Johnson methods.

## 📋 Overview

This project demonstrates how **PowerTransformer** improves linear regression models by transforming skewed features into normally distributed ones using a concrete strength dataset.

## 🎯 Problem

Predict concrete compressive strength based on mix ingredients and age.

## 📊 Dataset

**File**: `concrete_data.csv`

**Features** (8):
- Cement, Blast Furnace Slag, Fly Ash, Water
- Superplasticizer, Coarse Aggregate, Fine Aggregate, Age

**Target**: Strength (concrete compressive strength in MPa)

## 🛠️ Technologies

- Python 3.8
- pandas, numpy, matplotlib, seaborn
- scikit-learn (PowerTransformer, LinearRegression)
- scipy (for statistics)

## 🔑 Key Concepts

### What is Power Transformation?

Converts skewed distributions → Normal distributions for better model performance

### Two Methods Compared:

| Method | Use Case | Requirement |
|--------|----------|-------------|
| **Box-Cox** | Right-skewed data | All values must be > 0 |
| **Yeo-Johnson** | Any skewed data | Works with negative values too |

## 📈 Results

### Model Performance (R² Score):

| Approach | R² Score | Improvement |
|----------|----------|-------------|
| Without Transformation | 0.6155 | Baseline |
| With Box-Cox | 0.6268 | +1.8% |
| With Yeo-Johnson | 0.6314 | +2.6% 🎯 |

**Winner**: Yeo-Johnson transformation provides the best results!

## 🚀 Usage

```python
from sklearn.preprocessing import PowerTransformer

# Box-Cox (only for positive values)
pt_boxcox = PowerTransformer(method='box-cox')
X_transformed = pt_boxcox.fit_transform(X_train)

# Yeo-Johnson (works with any values)
pt_yeo = PowerTransformer(method='yeo-johnson')
X_transformed = pt_yeo.fit_transform(X_train)

# View lambda values
print(pt_yeo.lambdas_)
```

## 📊 Transformations Applied

**Lambda values** show how much each feature was transformed:

```
Feature              Box-Cox    Yeo-Johnson
Cement               0.170      0.174
Blast Furnace Slag   0.017      0.016
Fly Ash             -0.136     -0.161
Water                0.808      0.771
Superplasticizer     0.264      0.254
Coarse Aggregate     1.129      1.130
Fine Aggregate       1.831      1.783
Age                  0.002      0.020
```

## 🎓 When to Use Power Transformation?

✅ **Use when:**
- Data is highly skewed (not bell-shaped)
- Using linear models (Linear Regression, Logistic Regression)
- Features have different scales
- Want to improve model accuracy

❌ **Don't use when:**
- Using tree-based models (Random Forest, XGBoost) - they don't need it
- Data is already normally distributed
- You need interpretability (transformed features are harder to explain)

## 💡 Key Takeaways

1. **Skewed data hurts linear models** - Power transformation fixes this
2. **Yeo-Johnson is more flexible** - works with negative values
3. **Check before-after distributions** - visualize to confirm improvement
4. **Small improvements matter** - 2.6% R² boost is significant in production

## 📁 Project Structure

```
Power_Transformer/
├── Power_Transformer.ipynb    # Main notebook
├── concrete_data.csv          # Dataset
└── README.md                  # This file
```

## 🔍 Quick Example

```python
# Before transformation
skewness = X_train['Cement'].skew()  # High skewness
# After transformation
skewness_after = X_transformed['Cement'].skew()  # Near 0 (normal)

# Result: Better model performance!
```

## 👨‍💻 Author

**Your Name**
- GitHub: Wahab-300

---
