
from sklearn.ensemble import RandomForestClassifier


def forest(X,Y,**kwargs):
    model = RandomForestClassifier(**kwargs).fit(X,Y)
    quality = model.feature_importances_
    return  quality, model





