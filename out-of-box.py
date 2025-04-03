from data import evaluate_model, get_dataset, undersample
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def out_of_box():
    df = get_dataset()
    df = undersample(df)
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)

    y_pred = nb_model.predict(X_test)
    evaluate_model(y_test, y_pred, "out-of-box")


if __name__ == "__main__":
    out_of_box()
