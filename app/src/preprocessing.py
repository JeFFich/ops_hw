import pandas as pd
from category_encoders import TargetEncoder

path_data = './input/'
train = pd.read_csv(path_data + 'train.csv')

# Кодирование категориального через terget encoding
Targetenc = TargetEncoder().fit(X=train.pack, y=train.binary_target)

# Кодирование числовых признаков через медианы
features_medians = {"сумма": 0, "on_net": 0, "секретный_скор": 0}
for feature in ["сумма", "on_net", "секретный_скор"]:
    # запомним медианы
    features_medians[feature] = train[feature].median()


def preprocess(save_location: str) -> pd.DataFrame:
    """Предобработка входных данных"""
    test = pd.read_csv(save_location)
    
    for feature in ["сумма", "on_net", "секретный_скор"]:
        test[feature].fillna(features_medians[feature], inplace=True)
    test['pack'] = Targetenc.transform(X=test.pack)
    test = test[["client_id", "pack", "сумма", "on_net", "секретный_скор"]]
    
    return test
