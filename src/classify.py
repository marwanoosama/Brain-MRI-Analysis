from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_classifier(features, labels):
    """
    Train an SVM classifier on the extracted features.

    Splits data into 80% train / 20% test, scales features
    using StandardScaler, and trains an RBF-kernel SVM.

    Args:
        features: 2D array of shape (n_samples, n_features).
        labels: 1D array of labels (0 = normal, 1 = abnormal).

    Returns:
        Tuple of (clf, scaler, X_test, y_test, y_pred) where:
        - clf: trained SVM classifier
        - scaler: fitted StandardScaler
        - X_test: scaled test features
        - y_test: true test labels
        - y_pred: predicted test labels
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = SVC(kernel='rbf', probability=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return clf, scaler, X_test, y_test, y_pred
