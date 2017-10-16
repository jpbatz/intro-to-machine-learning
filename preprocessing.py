import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

DEFAULT_FILENAME = 'data.csv'


def preprocess(
        filename=DEFAULT_FILENAME,
        use_only_indexes=None,
        handle_missing_data=False,
        missing_data_index_range=(0, 0),
        handle_categorical_data_X=False,
        handle_categorical_data_y=False,
        categorical_data_X_index=0,
        scale_X=False,
        scale_y=False,
        test_set_size=0.2,
        test_set_random_seed=0,
        ):
    dataset = pd.read_csv(filename)

    X, y = break_dataset(dataset, use_only_indexes)

    if handle_missing_data:
        X = impute_missing_data(X, missing_data_index_range)

    if handle_categorical_data_X:
        X = encode_categorical_data_X(X, categorical_data_X_index)

    if handle_categorical_data_y:
        y = encode_categorical_data_y(y)

    # split into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_set_size,
        random_state=test_set_random_seed if test_set_random_seed else 0,
        )

    if scale_X:
        X_train, X_test = feature_scale_X(X_train, X_test)

    if scale_y:
        y_train = feature_scale_y(y_train)

    return X, y, X_train, y_train, X_test, y_test


def break_dataset(dataset, use_only_indexes=None):
    if use_only_indexes:
        X = dataset.iloc[:, use_only_indexes[0]:use_only_indexes[1]]
    else:
        X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    return X, y


def impute_missing_data(X, range):
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imputer = imputer.fit(X[:, range[0]:range[1]])
    X[:, range[0]:range[1]] = imputer.transform(X[:, range[0]:range[1]])
    return X


def encode_categorical_data_X(X, index):
    label_encoder_X = LabelEncoder()
    X[:, index] = label_encoder_X.fit_transform(X[:, index])
    one_hot_encoder = OneHotEncoder(categorical_features=[index])
    X = one_hot_encoder.fit_transform(X).toarray()
    return X


def encode_categorical_data_y(y):
    label_encoder_y = LabelEncoder()
    y = label_encoder_y.fit_transform(y)
    return y


def feature_scale_X(X_train, X_test):
    # feature scaling
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    if X_test.shape[0] > 0:
        X_test = scaler.transform(X_test)
    return X_train, X_test


def feature_scale_y(y_train):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    y_train = scaler.fit_transform(y_train)
    return y_train
