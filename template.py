!pip install pandas scikit-learn hmmlearn
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import accuracy_score

#  Load a sample dataset (Iris by default; change as needed)
# df  = pd.read_csv('dataset.csv')
# X = df.iloc[:, :-1].values
# y = df.iloc[:, -1].values


data = load_iris()
X, y = data.data, data.target

#Label encoding for categorical targets if necessary
if isinstance(y[0], str):
    le = LabelEncoder()
    y = le.fit_transform(y)


# Optional scaling for models like SVM, KNN, MLP
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

#  Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 1. Decision Tree (Gini)
# from sklearn.tree import DecisionTreeClassifier
# model = DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=0)

# 2. Support Vector Machine (SVM)
# from sklearn.svm import SVC
# model = SVC(kernel='rbf', C=1.0, gamma='scale')

# 3. MLPClassifier (Backpropagation)
# from sklearn.neural_network import MLPClassifier
# model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=300)

# 4 or 9. KMeans Clustering
# from sklearn.cluster import KMeans
# model = KMeans(n_clusters=3, init='k-means++', random_state=42)
# model.fit(X)
# print("KMeans labels:", model.labels_)

# 5. Random Forest (Gini)
# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=42)


# 7. Hidden Markov Model (Multinomial)
# from hmmlearn import hmm
# model = hmm.MultinomialHMM(n_components=3, n_iter=100)
# # NOTE: X must be reshaped as a sequence of integers and `lengths` parameter must be passed
# # Example:
# import numpy as np
# X_seq = np.random.randint(0, 3, size=(100, 1))
# model.fit(X_seq, lengths=[len(X_seq)])
# print(model.predict(X_seq))

# 8. ID3 (Simulated using Entropy)
# from sklearn.tree import DecisionTreeClassifier
# model = DecisionTreeClassifier(criterion='entropy')

# 11. Naive Bayes (Gaussian)
# from sklearn.naive_bayes import GaussianNB
# model = GaussianNB()

# 12. K-Nearest Neighbors (KNN)
# from sklearn.neighbors import KNeighborsClassifier
# model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

# 13. Sentiment Analysis (Naive Bayes + Text)
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.feature_extraction.text import CountVectorizer
# text_data = ["I love this product", "Worst app ever", "Amazing service", "Horrible experience"]
# labels = [1, 0, 1, 0]  # 1 = positive, 0 = negative
# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(text_data)
# model = MultinomialNB()
# model.fit(X, labels)
# print("Prediction:", model.predict(vectorizer.transform(["Very happy with the service"])))
# exit()

# 16. Email Spam Detection using HMM
# from hmmlearn import hmm
# model = hmm.MultinomialHMM(n_components=2, n_iter=100)
# Similar to #7 — Use preprocessed sequences

# 18. HMM for Diabetes (GaussianHMM)
# from hmmlearn import hmm
# model = hmm.GaussianHMM(n_components=3, covariance_type='full', n_iter=100)
# Similar input note: X should be 2D (samples, features) with real values


try:
    print(model)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Test Data:")
    print(X_test)
    print("Predictions:")
    print(y_pred)
    print("Accuracy:", accuracy_score(y_test, y_pred))
except Exception as e:
    print("NOTE: This model may need special preprocessing or input format.")
    print("Error:", e)

