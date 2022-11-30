import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from src.preprocessing import read_chat_file_pauses_to_dict, chat_count_dict_to_df
from src.classifier import split_train_test, fit_classifier, predict, eval_clf
from sklearn.preprocessing import MinMaxScaler


def read_data() -> pd.DataFrame:
    """
    Read the all the files.
    """
    # Dementia
    directory_path = "./chat_files/dementia/cookie"
    chat_file_pauses_dict = read_chat_file_pauses_to_dict(directory_path, filter_by_participant="PAR", label=1)
    df_dementia = chat_count_dict_to_df(chat_file_pauses_dict)

    # Control
    directory_path = "./chat_files/control/cookie"
    chat_file_pauses_dict = read_chat_file_pauses_to_dict(directory_path, filter_by_participant="PAR", label=0)
    df_control = chat_count_dict_to_df(chat_file_pauses_dict)

    # Concat dataframes
    df = pd.concat([df_dementia, df_control]).sort_values("file_path").reset_index(drop=True)


    # apply normalization techniques
    cols_to_norm = ['(.)', '(..)', '(...)', '&']
    scaler = MinMaxScaler()
    df[cols_to_norm] = scaler.fit_transform(df[cols_to_norm])

    return df


def traintest(df):
    X = df[["(.)", "(..)", "(...)", "&"]]
    y = df["label"]
    return split_train_test(X=X, y=y, test_size=0.2, random_state=42)


def dummy_classifier(df):
    """
    Make predictions that ignore the input features
    """
    X_train, X_test, y_train, y_test = traintest(df)

    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train, y_train)

    return dummy_clf.score(X_test, y_test)


def evaluate_classifier(df: pd.DataFrame, clf: Pipeline) -> dict:
    """
    Evaluate a classifier.
    """
    X_train, X_test, y_train, y_test = traintest(df)

    trained_clf = fit_classifier(clf, X_train, y_train)
    y_pred = predict(trained_clf, X_test)

    eval_scores = eval_clf(y_test, y_pred)
    return eval_scores


if __name__ == "__main__":
    df = read_data()

    print("\nBaseline: ", dummy_classifier(df), "\n")

    print("SVM: ", evaluate_classifier(df, SVC(kernel="linear")), "\n")

    print("Naive Bayes: ", evaluate_classifier(df, GaussianNB()), "\n")

    print("KNN:", evaluate_classifier(df, KNeighborsClassifier()))

    # grid search kunnen doen, bijv. n_neighbors (KNN), C (SVM), kernel (SVM), etc.
    #for kernel in ["rbf", "linear", "poly"]:
     #   for c in range(1, 100):
      #      scores = evaluate_classifier(df, SVC(kernel=kernel, C=(c / 1000)))
       #     print(f"SVM (kernel={kernel}, C={(c/1000)}). Scores:", scores)