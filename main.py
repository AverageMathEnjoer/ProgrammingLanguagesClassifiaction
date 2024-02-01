import Model
import DataPreparator
from catboost import CatBoostClassifier
import pandas as pd

if __name__ == '__main__':
    model = Model.Classifier(json_file_path="config.json")
    model.contest_solution("solution")




