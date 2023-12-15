# Data-Preparation-Group-4
## The problem statement
Many people struggle to get loans due to insufficient or non-existent credit histories. And, unfortunately, this population is often taken advantage of by untrustworthy lenders. Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. To make sure this underserved population has a positive loan experience, Home Credit makes use of a variety of alternative data including telco and transactional information to predict their clients' repayment abilities.

While Home Credit is currently using various statistical and machine learning methods to make these predictions, they're challenging the data science community to help them unlock the full potential of their data. Doing so will ensure that clients capable of repayment are not rejected and that loans are given with a principal, maturity, and repayment calendar that will empower their clients to be successful.

## Project Description
A simple notebook on Exploration (EDA) and Baseline Machine Learning Model of Home Credit default risk data to predict future payment problems for clients of the company.

## Project Structure
### Feature Engineering
`feature engineering`
  
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
`modelling_experiment`

  - `(submission_1)lgbm_gridsearchCV.py`: Script for LightGBM model training with grid search cross-validation.
  - `(submission_2)kfold_lgbm_gridsearchCV.py`: Script for LightGBM model training with k-fold cross-validation and grid search.
  - `VIF_gridsearchCV.py`: Script for feature selection using Variance Inflation Factor (VIF) and grid search.
  - `kfold_lgbm_optuna.py`: Script for LightGBM model training with k-fold cross-validation using Optuna.
  - `selectk_gridsearchCV.py`: Script for feature selection using SelectKBest and grid search.
  - `utils.py`: Contains utility functions specific to the modeling experiments.

## Installation




## Usage


## Contributing
  * Trinh Thi Minh Tam
    - EDA<br /> 
      * Review and fix comments for all EDA files
      * Refactor Code 
    - Feature Engineer <br />
    - Tunning model <br />
  * Nguyen Manh Hung 
    - EDA<br />
      * Bureau
      * Bureau_balance
    - Slide <br />
  * Nguyen Ha Phuong
    - EDA<br />
      * Application_train 
      * Instalment_payment
      * Credit_card_balance 
      * POS_CASH_balance
      * Previous_application


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


