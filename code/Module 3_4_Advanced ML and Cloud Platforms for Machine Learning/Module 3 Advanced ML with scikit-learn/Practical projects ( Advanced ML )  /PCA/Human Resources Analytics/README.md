# Employee Clustering Analysis Using HR Analytics


## Project Overview

This project analyzes employee data to identify patterns and segments within the workforce using advanced analytics techniques. By applying dimensionality reduction through Principal Component Analysis (PCA) and clustering algorithms, the analysis uncovers meaningful groups of employees that share similar characteristics. These insights can inform HR strategies related to retention, engagement, and talent management.

## Dataset

The analysis uses the [HR Analytics Dataset](https://www.kaggle.com/ludobenistant/hr-analytics) from Kaggle, containing information related to:

- Employee demographics (age, gender, marital status)
- Job-related factors (department, role, level, income)
- Satisfaction metrics (job satisfaction, work-life balance)
- Tenure information (years at company, years in role)
- Attrition status

## Key Features

- **Comprehensive Data Exploration**: In-depth analysis of HR data including correlation studies, distribution analysis, and feature importance evaluation
- **Advanced Visualization**: Interactive and informative visualizations highlighting key HR insights
- **Dimensionality Reduction**: Principal Component Analysis (PCA) to identify underlying patterns
- **Employee Segmentation**: Clustering techniques to identify distinct employee groups

## Key Insights

### Attrition Risk Factors

- **Highest attrition** occurs among:

  - Young employees (25-35 years old)
  - Employees with lower monthly income
  - Those with fewer years at the company (0-3 years)
  - Single employees working overtime
  - Staff in Human Resources, Technical, and Marketing fields
- **Work-Life Balance Impact**: Poor work-life balance combined with low job satisfaction leads to 47.1% attrition rate

### Feature Importance

- **Overtime** is the strongest predictor of attrition
- **Marital status** (single) strongly correlates with leaving
- **Tenure-related** variables are significant determinants
- **Career progression metrics** form the first principal component
- **Compensation and performance ratings** form the second principal component

## Technologies Used

- **Python**: Core programming language
- **Data Analysis**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Dimensionality Reduction**: PCA
- **Clustering**: K-means

## Getting Started

### Prerequisites

- Python 3.8+
- Required libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

### Installation

```bash
# Clone the repository
git clone https://github.com/username/employee-clustering-analysis.git
cd employee-clustering-analysis

# Create and activate virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis

The analysis is contained in a Jupyter Notebook:

```bash
jupyter notebook main.ipynb
```

## Key Visualizations

- Distribution of key features by attrition status
- Correlation heatmap of numerical features
- Feature importance for predicting attrition
- PCA visualization of employee segments
- Cluster analysis with demographic profiling

## Business Recommendations

1. **Target retention strategies** for younger employees (25-35) with lower incomes
2. **Review overtime policies** to prevent burnout and increase retention
3. **Enhance early-career support** through mentorship and growth opportunities
4. **Monitor job satisfaction** and work-life balance, especially for at-risk groups
5. **Develop career paths** for entry-level employees to improve retention

## Future Work

- Implement predictive modeling for attrition risk scoring
- Develop interactive dashboard for HR decision-makers
- Incorporate time-series analysis for temporal patterns
- Compare clustering results with supervised learning approaches
- Add external market data for competitive analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

[Your Name] - [@your_twitter](https://twitter.com/your_twitter) - email@example.com

Project Link: [https://github.com/username/employee-clustering-analysis](https://github.com/username/employee-clustering-analysis)
