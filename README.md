# Linear_Regression_From_Scratch
This is a practice file inspired by Venelin Valkov and his repository of Machine Learning from Scratch which you can find https://github.com/curiousily/Machine-Learning-from-Scratch. This is my own take on how to get the same result with a different way. This is purely for practice and revision purposes and all credits go to Venelin Valkov for such a inspiring repository! 

# House Price Prediction with Linear Regression

A comprehensive implementation of Linear Regression from scratch using Python, featuring interactive visualizations and gradient descent optimization for predicting house sale prices.

## Project Overview

This project demonstrates the complete machine learning pipeline for house price prediction, including:

- **Custom Linear Regression Implementation** - Built from scratch using gradient descent
- **Data Exploration & Visualization** - Comprehensive EDA with correlation analysis
- **Feature Engineering** - Data standardization and preprocessing
- **Model Training Animation** - Visual representation of gradient descent optimization
- **Performance Evaluation** - Cost function tracking and model validation

## 🎯 Key Features

- ✅ **From-Scratch Implementation** - No sklearn dependency for core algorithm
- ✅ **Interactive Visualizations** - Animated gradient descent process
- ✅ **Comprehensive EDA** - Correlation heatmaps, distribution plots, and pairplots
- ✅ **Unit Testing** - Automated tests for cost function and model accuracy
- ✅ **Multiple Regression Models** - Single and multivariate implementations

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/MushiNizi74/Linear_Regression.git
cd house-price-linear-regression
```

2. Download the dataset:
   - Place `train.csv` and `test.csv` in a `data/` folder
   - Update file paths in the notebook accordingly

3. Run the Jupyter notebook:
```bash
jupyter notebook linear_regression_analysis.ipynb
```

## 📁 Project Structure

```
house-price-linear-regression/
│
├── linear_regression.ipynb    # Main analysis notebook
├── data/                              # Dataset folder
│   ├── train.csv                      # Training data
│   └── test.csv                       # Test data
├── images/                            # Generated plots and animations
├── README.md                          # This file
└── requirements.txt                   # Python dependencies
```

## 🔍 What You'll Learn

### 1. **Mathematical Foundations**
- Linear regression theory and assumptions
- Gradient descent optimization
- Cost function (Mean Squared Error) implementation
- Feature standardization using Z-score normalization

### 2. **Data Science Workflow**
- Exploratory Data Analysis (EDA)
- Missing data analysis
- Feature correlation and selection
- Data preprocessing and scaling

### 3. **Visualization Techniques**
- Distribution plots and scatter plots
- Correlation heatmaps
- Box plots for categorical features
- Animated training process visualization

### 4. **Programming Best Practices**
- Object-oriented programming for ML models
- Unit testing for algorithm validation
- Clean, documented code structure
- Reproducible analysis with random seeds

## 📈 Model Performance

The implementation achieves comparable results to sklearn's LinearRegression:

- **Single Feature Model** (Living Area only):
  - Converges to optimal weights: `[180921.20, 56294.90]`
  - Demonstrates clear linear relationship visualization

- **Multiple Feature Model** (Quality + Living Area + Garage):
  - Enhanced prediction accuracy with multiple features
  - Lower final cost compared to single feature model

## 🎨 Visualizations

### Data Exploration
- **Distribution Analysis**: Sale price distribution with KDE
- **Scatter Plots**: Feature vs target relationships
- **Correlation Heatmaps**: Feature correlation analysis
- **Box Plots**: Categorical feature analysis

### Model Training
- **Animated Gradient Descent**: Watch the model learn in real-time
- **Cost Function Convergence**: Track training progress
- **Regression Line Evolution**: See how predictions improve

## 🧪 Testing

The project includes comprehensive unit tests:

```python
# Run tests
execute_unit_tests()
```

Tests cover:
- Cost function accuracy
- Model weight convergence
- Edge cases and boundary conditions

## 📊 Key Insights

1. **Living Area** shows strong positive correlation with sale price
2. **Overall Quality** is a crucial predictor for house values
3. **Garage Cars** capacity correlates with higher-priced homes
4. **Gradient Descent** successfully converges to optimal solution
5. **Feature Standardization** is essential for proper convergence

## 🔧 Customization

### Modify Learning Parameters
```python
# Adjust learning rate and iterations
model.train_model(features, targets, iterations=5000, learning_rate=0.001)
```

### Add New Features
```python
# Include additional features
new_features = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars']
```

### Change Animation Settings
```python
# Customize animation speed and frames
animation.FuncAnimation(fig, animate, frames=500, interval=20)
```

## 🙏 Acknowledgments

- Dataset: House Prices - Advanced Regression Techniques (Kaggle)
- Inspiration: Andrew Ng's Machine Learning Course
- curiousily for the inspiration
