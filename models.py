import xgboost as xgb
import sklearn.ensemble as sk
import optuna

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from metrics import *
from sklearn.metrics import f1_score

#######################################################################################
#######################################################################################

class XGBClassifier:

    def __init__(self, features=[], params={}):
        self.attribute()
        self.params = dict(self.init_params)
        self.params.update(params)
        self.features = list(features) if features else self.features_set
        self.estimator = xgb.XGBClassifier(**params)


    def attribute(self):
        self.init_params = { }
        self.features_set = [
            'c1',
            'c2',
            'c3',
            'c6',
            # 'c13',
            'c15',
            'c18',
            'c21',
            'c23',
            'c24',
            'c25',
        ]

        self.target = 'c32'

    def find_best_params(self, DataSet, test_size=0.25, n_trials=5):
        X, y = DataSet.loc[:, self.features], DataSet.loc[:, self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        self.le = LabelEncoder()
        y_train = self.le.fit_transform(y_train)
        y_test = y_test.to_numpy()

        def objective(trial):
            learning_rate = trial.suggest_float('learning_rate', 0.001, 0.01)
            n_estimators = trial.suggest_int('n_estimators', 10, 1000)
            max_depth = trial.suggest_int('max_depth', 1, 200)

            params = {
                'learning_rate': learning_rate,
                'n_estimators': n_estimators,
                'max_depth': max_depth,
            }
            self.estimator.set_params(**params)
            self.estimator.fit(X_train, y_train)

            y_pred = self.le.inverse_transform(self.estimator.predict(X_test))
            return f1_score(y_test, y_pred, average='weighted')

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        self.best_params = study.best_params
        self.best_value = study.best_value
        self.estimator.set_params(**self.best_params)


    def fit(self, X_train, y_train):
        self.le = LabelEncoder()
        self.X_train = X_train
        self.y_train = self.le.fit_transform(y_train)
        self.estimator.fit(self.X_train, self.y_train)

    def predict(self, X_test):
        return self.le.inverse_transform(self.estimator.predict(X_test))


#######################################################################################
#######################################################################################

class RandomForestClassifier:

    def __init__(self, features=[], params={}):
        self.attribute()
        self.params = dict(self.init_params)
        self.params.update(params)
        self.features = list(features) if features else self.features_set
        self.estimator = sk.RandomForestClassifier(**params)


    def attribute(self):
        self.init_params = {
            'n_estimators': 808,
            'max_depth': 105,
            'max_features': 8
        }
        self.features_set = [
            'c1',
            'c2',
            'c3',
            'c6',
            # 'c13',
            'c15',
            'c18',
            'c21',
            'c23',
            'c24',
            'c25',
        ]

        self.target = 'c32'

    def find_best_params(self, DataSet, test_size=0.25, n_trials=5):
        X, y = DataSet.loc[:, self.features], DataSet.loc[:, self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        self.le = LabelEncoder()
        y_train = self.le.fit_transform(y_train)
        y_test = y_test.to_numpy()

        def objective(trial):
            # criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
            n_estimators = trial.suggest_int('n_estimators', 10, 1000)
            max_depth = trial.suggest_int('max_depth', 1, 200)
            max_features = trial.suggest_int('max_features', 1, 10)

            params = {
                # 'criterion': criterion,
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'max_features': max_features
            }
            self.estimator.set_params(**params)
            self.estimator.fit(X_train, y_train)

            y_pred = self.le.inverse_transform(self.estimator.predict(X_test))
            return f1_score(y_test, y_pred, average='weighted')

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        self.best_params = study.best_params
        self.best_value = study.best_value
        self.estimator.set_params(**self.best_params)


    def fit(self, X_train, y_train):
        self.le = LabelEncoder()
        self.X_train = X_train
        self.y_train = self.le.fit_transform(y_train)
        self.estimator.fit(self.X_train, self.y_train)

    def predict(self, X_test):
        return self.le.inverse_transform(self.estimator.predict(X_test))
