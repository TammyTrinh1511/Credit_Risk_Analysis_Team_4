# Data-Preparation-Group-4

### This is the notebook for EDA project by Group 4, DSEB K63, NEU
  
  * Trinh Thi Minh Tam
  * Nguyen Ha Phuong
  * Nguyen Manh Hung


## Project Structure
### EDA
- `eda_application.ipynb`: Jupyter Notebook for exploratory data analysis on the application data.
- `bureau_balance_analysis.ipynb`: Jupyter Notebook for analyzing bureau balance data.
- `previous_application.ipynb`: Jupyter Notebook for exploring previous application data.
- `POS_CASH_analysis.ipynb`: Jupyter Notebook for analyzing POS CASH data.
- `installment_payment.ipynb`: Jupyter Notebook for installment payment analysis.
- `credit_card_balance.ipynb`: Jupyter Notebook for credit card balance analysis.
### Feature Engineering

- `application_pipeline.py`: Handles the application data processing pipeline.
- `bureau_pipeline.py`: Manages the bureau data processing pipeline.
- `config.py`: Configuration file containing project-wide settings.
- `credit_card_pipeline.py`: Manages the credit card data processing pipeline.
- `installments_payments_pipeline.py`: Deals with the installments payments data processing pipeline.
- `merge_data.py`: Script for merging different datasets.
- `pos_cash_pipeline.py`: Manages the point of sale (POS) cash data processing pipeline.
- `previous_application_pipeline.py`: Manages the previous application data processing pipeline.
- `utils.py`: Contains utility functions used across different modules.

### Modeling

- `(submission_1)lgbm_gridsearchCV.py`: Script for LightGBM model training with grid search cross-validation.
- `(submission_2)kfold_lgbm_gridsearchCV.py`: Script for LightGBM model training with k-fold cross-validation and grid search.
- `VIF_gridsearchCV.py`: Script for feature selection using Variance Inflation Factor (VIF) and grid search.
- `kfold_lgbm_optuna.py`: Script for LightGBM model training with k-fold cross-validation using Optuna.
- `selectk_gridsearchCV.py`: Script for feature selection using SelectKBest and grid search.
- `utils.py`: Contains utility functions specific to the modeling experiments.

## Installation

Clone về và tạo virtual environment
```
git clone TammyTrinh1511/Credit_Risk_Analysis_Team_4
conda create -n credit-risk
conda activate credit-risk # mỗi khi muốn dùng thì activate 
conda install --file requirements.txt  # Nếu k dc thì python -m pip install -r requirements.txt 

```
Muốn upload code
```
git pull 
git add myfile 
git commit -m "add ..." 
git push
```

## Useage



## Contributing
  * Trinh Thi Minh Tam


  * Nguyen Ha Phuong


  * Nguyen Manh Hung


