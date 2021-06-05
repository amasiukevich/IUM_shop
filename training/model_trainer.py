import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

from constants import TARGET_FEATURE_NAME, SCORING_NAME

from sklearn.model_selection import GridSearchCV, train_test_split, KFold


# models
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from xgboost import XGBRegressor


# A place for custom classifier - Rolling average for N months
def train_baseline_model():
    pass

# end of that place
def perform_last_month_split(feature_df, train_index, test_index):
    pass


def train_best_model(feature_df: pd.DataFrame):

    # TODO: selecting the best 6 features (according to the ranking of mutual information score)

    best_features = [
        "buying_sessions_prev_month",
        "buying_sessions_MA3",
        "buying_ratio_prev_month",
        "money_monthly_MA3",
        "buying_ratio_MA3",
        "prev_month_spendings"
    ]


    X = feature_df[best_features]
    y = feature_df[TARGET_FEATURE_NAME]

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, shuffle=True)


    k_fold = KFold(n_splits=10).split(train_X, train_y)

    models = [
        train_lin_reg(train_X, train_y, kfold=k_fold),
        train_lasso(train_X, train_y, kfold=k_fold),
        train_ridge(train_X, train_y, kfold=k_fold),
        train_svm(train_X, train_y, kfold=k_fold),
        train_rf(train_X, train_y, kfold=k_fold),
        train_adaboost(train_X, train_y, kfold=k_fold),
        train_xgboost(train_X, train_y, kfold=k_fold)
    ]

    best_model = None
    best_score = float('inf')

    for model in models:

        preds = model.predict(test_X)
        mae = mean_absolute_error(test_y, preds)

        if mae < best_score:
            best_model = model
            best_score = mae


    print("Best score is: ", best_score)
    print("Best model is: ", best_model)


    names = {
        "linear_regression": LinearRegression,
        "lasso": Lasso,
        "ridge": Ridge,
        "svm": SVR,
        "random_forest": RandomForestRegressor,
        "adaboost": AdaBoostRegressor,
        "xgboost": XGBRegressor
    }

    model_name = None
    for name, _class in names.items():
        if isinstance(best_model, _class):
            model_name = name
            break

    return best_model, model_name


# Linear
def train_lin_reg(X, y, kfold):

    print("\nTraining Linear Regression")
    lin_reg = LinearRegression()

    params = {
        "fit_intercept": [True, False]
    }

    best_lin_reg, best_score = train_model(lin_reg, params, X, y, kfold)
    print(f"Best linear regression score: {best_score}")

    return best_lin_reg


def train_lasso(X, y, kfold):

    print("\nTraining Lasso")
    model_lasso = Lasso()

    params = {
        "alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        'fit_intercept': [True, False]
    }

    best_lasso, best_score = train_model(model_lasso, params, X, y, kfold)
    print(f"Best Lasso score: {best_score}")

    return best_lasso


def train_ridge(X, y, kfold):

    print("\nTraining Ridge")
    model_ridge = Ridge()

    params = {
        "alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        'fit_intercept': [True, False]
    }

    best_ridge, best_score = train_model(model_ridge, params, X, y, kfold)
    print(f"Best Ridge score: {best_score}")

    return best_ridge


def train_svm(X, y, kfold):

    print("\nTraining SVM")
    model_svm = SVR()

    params = {
        "kernel": ['rbf'],
        "C": [1000,3000, 5000, 7000, 8000, 9000, 10000, 15000]
    }

    best_svm, best_score = train_model(model_svm, params, X, y, kfold)
    print(f"Best SVM score is {best_score}")

    return best_svm


def train_rf(X, y, kfold):

    print("\nTraining Random Forest")
    model_rf = RandomForestRegressor()

    params = {
        "n_estimators": [11, 17, 25, 31],
        "min_samples_split": [2, 3, 5, 7, 8, 10, 13],
        "max_depth": [2, 3, 5, 7, 9],
        "criterion": ["mae"]
    }

    best_rf, best_score = train_model(model_rf, params, X, y, kfold)
    print(f"Best Random Forest score: {best_score}")

    return best_rf


def train_adaboost(X, y, kfold):

    print("\nTraining AdaBoost")

    model_ada_boost = AdaBoostRegressor()

    svm_base = SVR(kernel='rbf')
    tree_base = DecisionTreeRegressor(max_depth=5)

    params = {
        "n_estimators": [2, 3, 5, 8, 13, 21],
        "base_estimator": [svm_base, tree_base],
        "learning_rate": [0.001, 0.01, 0.1]
    }

    best_ada_boost, best_score = train_model(model_ada_boost, params, X, y, kfold)
    print(f"Best AdaBoost score: {best_score}")

    return best_ada_boost



def train_xgboost(X, y, kfold):

    print("Training XGBoost")
    model_xgb = XGBRegressor()
    params = {
        'booster': ['gbtree'],
        'verbosity': [0],
        'learning_rate': [0.001, 0.01, 0.1],
        'gamma': [1000, 10000, 20000],
        'max_depth': [5],
        'lambda': [0.5, 1],
        'alpha': [0.3, 0.5, 0.7, 1]
    }

    best_xgb, best_score = train_model(model_xgb, params, X, y, kfold)

    print(f"Best XGBoost Score: {best_score}")
    return best_xgb



def train_model(model, params, X, y, kfold):

    kfold = KFold(n_splits=10).split(X, y)
    gs = GridSearchCV(
        model,
        param_grid=params,
        scoring=SCORING_NAME,
        cv=kfold
    )

    gs.fit(X, y)

    print("\n", gs.best_params_)

    best_model = gs.best_estimator_
    best_score = gs.best_score_

    return best_model, (-1) * best_score