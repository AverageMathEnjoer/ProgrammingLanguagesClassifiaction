from catboost import CatBoostClassifier
from joblib import dump, load
import pandas as pd
import os.path
import DataPreparator


class Classifier(CatBoostClassifier):
    def __init__(self, json_file_path="config.json", **kwds):
        self.contest_data = DataPreparator.Data(json_file_path)
        super().__init__(**kwds)
        if os.path.exists(self.contest_data.cnf["model_path"]) and os.path.exists(self.contest_data.cnf["vec_path"]):
            print("Model find")
            self.load_model(self.contest_data.cnf["model_path"])
            self.contest_data.vectorizer = load(self.contest_data.cnf["vec_path"])
        else:
            print("Model not find, start training")
            self._dfit()
            dump(self.contest_data.vectorizer, self.contest_data.cnf["vec_path"])
            self.save_model(self.contest_data.cnf["model_path"])

    def contest_solution(self, path="output"):
        mask = pd.read_csv(self.contest_data.cnf["data_path"] + self.contest_data.cnf["test_ids"], index_col=0)
        prediction = self.predict(self.contest_data.return_test())
        mask['label'] = prediction.T[0].tolist()
        mask.reset_index(drop=True).to_csv(f"{path}.csv", index=False)

    def _dfit(self):
        d = self.contest_data.transform_data()
        self.fit(d[0][0], d[0][1], eval_set=d[1], verbose=False, plot=False)
    '''
    Функция для распознавания произвольного файла
    '''
    def analyze(self, path):
        translator = ["C#", "C++", "F#", "Haskell", "Java", "Kotlin", "R"]
        with open(path) as f:
            s = f.read()
        return translator[int(self.predict(self.contest_data.vectorizer.transform([s])[0][0]))]
