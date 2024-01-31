import DataPreparator

if __name__ == '__main__':
    text_base = DataPreparator.Data("config.json")
    a = text_base.transform_data()
    print(text_base.test)


