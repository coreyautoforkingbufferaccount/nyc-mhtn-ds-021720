import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier

#A python file that contains various functions for single use.

def plot_knn_f1s(X_train, X_test, y_train, y_test):
    k_scores=[]
    f_scores = []
    k_range = list(range(1, 20))
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred=knn.predict(X_test)
        acc=metrics.accuracy_score(y_test,y_pred)
        f1 = metrics.f1_score(y_test,y_pred)
        k_scores.append(acc)
        f_scores.append(f1)
    list(zip(k_scores, f_scores))
    sns.set_style("darkgrid")
    plt.figure(figsize=(12, 6))
    plt.plot(k_range, f_scores, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('F1 score by K Value')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy Score')
    plt.show()


