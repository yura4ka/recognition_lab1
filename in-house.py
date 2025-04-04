from data import evaluate_model, get_dataset, undersample
from sklearn.model_selection import train_test_split
import numpy as np


class NaiveBayesClassifier:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}

        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0) + 1e-6
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def _gaussian_likelihood(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = -0.5 * ((x - mean) ** 2) / var
        denominator = -0.5 * np.log(2 * np.pi * var)
        return numerator + denominator

    def _log_posterior(self, x):
        posteriors = []
        for c in self.classes:
            prior = np.log(self.priors[c])
            class_conditional = np.sum(self._gaussian_likelihood(c, x))
            posterior = prior + class_conditional
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        return np.array([self._log_posterior(x) for x in X])


def in_house():
    df = get_dataset()
    df = undersample(df)
    X = df.drop(columns=["Outcome"]).values
    y = df["Outcome"].values
    random_state = 8323

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    nb_model = NaiveBayesClassifier()
    nb_model.fit(X_train, y_train)

    y_pred = nb_model.predict(X_test)
    evaluate_model(y_test, y_pred, "undersampled-in-house")


if __name__ == "__main__":
    in_house()
