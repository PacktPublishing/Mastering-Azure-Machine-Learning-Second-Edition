import numpy as np
import argparse
import matplotlib.pyplot as plt
import lightgbm as lgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import joblib
from azureml.core import Dataset, Run

# Setup Run
# ---------------------------------------

# Load the current run and ws
run = Run.get_context()
ws = run.experiment.workspace


# Parse parameters
# ---------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str)
parser.add_argument('--boosting', type=str, default='dart')
parser.add_argument('--learning-rate', type=float, default=0.001)
parser.add_argument('--drop-rate', type=float, default=0.15)

parser.add_argument('--num-boost-round', type=int, default=500)
parser.add_argument('--early-stopping-rounds', type=int, default=200)
parser.add_argument('--min-data-in-leaf', type=int, default=20)
parser.add_argument('--feature-fraction', type=float, default=0.7)
parser.add_argument('--num-leaves', type=int, default=40)
args = parser.parse_args()

lgbm_params = {
    'application': 'binary',
    'metric': 'binary_logloss',
    'learning_rate': args.learning_rate,
    'boosting': args.boosting,
    'drop_rate': args.drop_rate,
    'min_data_in_leaf': args.min_data_in_leaf,
    'feature_fraction': args.feature_fraction,
    'num_leaves': args.num_leaves,
}


# Loading data
# ---------------------------------------

# Get a dataset by id
dataset = Dataset.get_by_id(ws, id=args.data)

# Load a TabularDataset into pandas DataFrame
df = dataset.to_pandas_dataframe()

y = df.pop('Survived')

# Split into training and validation set
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

categorical_features = ['Alone', 'Sex', 'Pclass', 'Embarked']

# Create an LGBM dataset for training
train_data = lgbm.Dataset(data=X_train, label=y_train, categorical_feature=categorical_features, free_raw_data=False)

# Create an LGBM dataset from the test
test_data = lgbm.Dataset(data=X_test, label=y_test, categorical_feature=categorical_features, free_raw_data=False)


# Logging
# ---------------------------------------
def azure_ml_callback(run):
    def callback(env):
        if env.evaluation_result_list:
            for data_name, eval_name, result, _ in env.evaluation_result_list:
                run.log("%s (%s)" % (eval_name, data_name), result)
    callback.order = 10
    return callback

def log_importance(clf, run):
    fig, ax = plt.subplots(1, 1)
    lgbm.plot_importance(clf, ax=ax)
    run.log_image("feature importance", plot=fig)

def log_params(params):
    for k, v in params.items():
        run.log(k, v)
    
def log_metrics(clf, X_test, y_test, run):
    preds = np.round(clf.predict(X_test))
    run.log("accuracy (test)", accuracy_score(y_test, preds))
    run.log("precision (test)", precision_score(y_test, preds))
    run.log("recall (test)", recall_score(y_test, preds))
    run.log("f1 (test)", f1_score(y_test, preds))


# Register model
# ---------------------------------------

def log_model(clf):
    joblib.dump(clf, 'lgbm.pkl')
    run.upload_file('lgbm.pkl', 'lgbm.pkl')
    run.register_model(model_name='lgbm_titanic', model_path='lgbm.pkl')


# Train
# ---------------------------------------
evaluation_results = {}
clf = lgbm.train(train_set=train_data,
                 params=lgbm_params,
                 valid_sets=[train_data, test_data], 
                 valid_names=['train', 'val'],
                 evals_result=evaluation_results,
                 num_boost_round=args.num_boost_round,
                 early_stopping_rounds=args.early_stopping_rounds,
                 verbose_eval=20,
                 callbacks = [azure_ml_callback(run)])

log_metrics(clf, X_test, y_test, run)
log_importance(clf, run)
log_model(clf)
log_params(lgbm_params)