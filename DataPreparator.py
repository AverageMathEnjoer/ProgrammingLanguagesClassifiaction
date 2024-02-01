import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import sklearn
from tqdm import tqdm


class Data:
    vectorizer: TfidfVectorizer

    def __init__(self, json_file_path="config.json"):
        with open(json_file_path) as conf:
            self.cnf = json.load(conf)
        self.vectorizer = TfidfVectorizer(sublinear_tf=True)
        self.data = pd.read_csv(self.cnf["data_path"] + self.cnf["train_labels"])
        self.test = pd.read_csv(self.cnf["data_path"] + self.cnf["test_ids"])

    def __read_data(self):
        for i in tqdm(range(len(self.data['id'])), desc="Loading training set"):
            with open(self.cnf["data_path"] + self.cnf["train_data"] + self.data['id'][i]) as file:
                s = file.read()
                self.data['id'][i] = s

    def __read_test(self):
        for i in tqdm(range(len(self.test['id'])), desc="Loading testing set"):
            with open(self.cnf["data_path"] + self.cnf["test_data"] + self.test['id'][i]) as file:
                s = file.read()
                self.test['id'][i] = s

    def transform_data(self):
        self.__read_data()
        x_train, x_val, y_train, y_val = train_test_split(self.data['id'].values, self.data['label'].values,
                                                          test_size=0.2, stratify=self.data['label'].values)
        self.vectorizer.fit(self.data['id'].values)
        train_vectors = self.vectorizer.transform(x_train)
        val_vectors = self.vectorizer.transform(x_val)
        return (train_vectors, y_train), (val_vectors, y_val)

    def return_test(self):
        self.__read_test()
        try:
            return self.vectorizer.transform(self.test['id'])
        except sklearn.exceptions.NotFittedError:
            print("Fitting forgiven")
            train = self.transform_data()
            return self.vectorizer.transform(self.test['id'])
