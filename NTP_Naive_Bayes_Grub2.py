# Made by Ömer Avcı XD Semai Miraç Arıcı Yunus Emre Çağan Buğra Karaahmetoğlu Sümeyye Üstünsoy

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("breast_cancer.csv")

data['Class'] = data['Class'].map({2: 'benign', 4: 'malignant'})

X = data.drop(['Sample code number', 'Class'], axis=1)
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#-------------------------------------
# Bernoulli Naive Bayes sınıflandırıcı =Multinominal naive bayes'e benzer şekil veriyor. ancak tahminler sadece boolean(ikili) şeklindedir.

bnb_classifier = BernoulliNB(binarize=5)
bnb_classifier.fit(X_train, y_train)

bnb_y_pred = bnb_classifier.predict(X_test)

accuracy = accuracy_score(y_test, bnb_y_pred)
precision = precision_score(y_test, bnb_y_pred, pos_label='malignant')
recall = recall_score(y_test, bnb_y_pred, pos_label='malignant')
f1 = f1_score(y_test, bnb_y_pred, pos_label='malignant')

print("\nBernoulliNB")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

cm = confusion_matrix(y_test, bnb_y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['benign', 'malignant'], yticklabels=['benign', 'malignant'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Bernoulli Confusion Matrix')

#------------------------------------
# Guassian Naive Bayes sınıflandırıcı = Bu yöntem eğitim verisinden her sınıf için ortalama(mean) ve standart sapma(standart deviation) değerleri tahmin edilir. Bu sayede dağılım Özetlenir.Gaussian Naive Bayes algoritmasında her sınıfın olasılıklarına ek olarak her sınıfın ortalama ve standart sapma değerleri de saklanır.

gnb_classifier = GaussianNB()
gnb_classifier.fit(X_train, y_train)
guassian_y_pred = gnb_classifier.predict(X_test)

accuracy = accuracy_score(y_test, guassian_y_pred)
precision = precision_score(y_test, guassian_y_pred, pos_label='malignant')
recall = recall_score(y_test, guassian_y_pred, pos_label='malignant')
f1 = f1_score(y_test, guassian_y_pred, pos_label='malignant')

print("\nGuassianNB")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

cm = confusion_matrix(y_test, guassian_y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['benign', 'malignant'], yticklabels=['benign', 'malignant'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Guassian Confusion Matrix')


#------------------------------------
# Multinomial Naive Bayes sınıflandırıcı = Multinomial naive Bayes'in sıklıkla kullanıldığı önemli bir alan, özelliklerin sınıflandırılacak belgelerdeki kelime sayımları veya frekansları ile ilişkili olduğu metin sınıflandırmasıdır.

mnb_classifier = MultinomialNB()
mnb_classifier.fit(X_train, y_train)
multinomial_y_pred = mnb_classifier.predict(X_test)

accuracy = accuracy_score(y_test, multinomial_y_pred)
precision = precision_score(y_test, multinomial_y_pred, pos_label='malignant')
recall = recall_score(y_test, multinomial_y_pred, pos_label='malignant')
f1 = f1_score(y_test, multinomial_y_pred, pos_label='malignant')

print("\nMultinomialNB")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

cm = confusion_matrix(y_test, multinomial_y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['benign', 'malignant'], yticklabels=['benign', 'malignant'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Multinomial Confusion Matrix')

plt.show()
