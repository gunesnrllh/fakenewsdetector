
Import of general libraries

add Codeadd Markdown
import numpy as np 
import pandas as pd
add Codeadd Markdown
Veriyi okuma işlemi

add Codeadd Markdown
data_true = pd.read_csv("../input/fake-and-real-news-dataset/True.csv")
data_false = pd.read_csv("../input/fake-and-real-news-dataset/Fake.csv")
add Codeadd Markdown
her satırın etiketlenmesi
add Codeadd Markdown
her satırın etiketlenmesi

add Codeadd Markdown
data_true["label"]= 0
data_false["label"]= 1
add Codeadd Markdown
Verilere bir bakış

add Codeadd Markdown
data_true.head()
add Codeadd Markdown
data_false.head()
add Codeadd Markdown
print(data_true.shape)
print(data_false.shape)
add Codeadd Markdown
veriler hakkında bilgi ediniyoruz

add Codeadd Markdown
data_true.info()
add Codeadd Markdown
data_false.info()
add Codeadd Markdown
Verilerin bir araya getirilmesi

add Codeadd Markdown
data = pd.concat([data_true, data_false])
data.shape
add Codeadd Markdown
from sklearn.utils import shuffle
data = shuffle(data)
data = data.reset_index(drop=True)
add Codeadd Markdown
data.head(10)
add Codeadd Markdown
y = data["label"]
y.shape
add Codeadd Markdown
CountVectorizer'ı kopyalanan verilerle kullanma

add Codeadd Markdown
from sklearn.feature_extraction.text import CountVectorizer
add Codeadd Markdown
copied_data = data.copy()
add Codeadd Markdown
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(copied_data)):
    review = re.sub('[^a-zA-Z]', ' ', copied_data['title'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
add Codeadd Markdown
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, ngram_range=(1,3))
X = cv.fit_transform(corpus).toarray()
add Codeadd Markdown
X.shape
add Codeadd Markdown
y = copied_data['label']
add Codeadd Markdown
Test Eğitimi Bölümü

add Codeadd Markdown
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=2)
add Codeadd Markdown
cv.get_feature_names()[:10]
add Codeadd Markdown
cv.get_params()
add Codeadd Markdown
cout_df = pd.DataFrame(X_train, columns=cv.get_feature_names())
add Codeadd Markdown
cout_df.head()
add Codeadd Markdown
import matplotlib.pyplot as plt
add Codeadd Markdown
Function for ploting
add Codeadd Markdown
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
​
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
​
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
​
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
add Codeadd Markdown
Use of Multinomial NB Classifier

add Codeadd Markdown
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import itertools
classifier = MultinomialNB()
add Codeadd Markdown
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy: %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['Fake', 'Real'])
add Codeadd Markdown
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
score
add Codeadd Markdown
Use of Passive Aggressive Classifier

add Codeadd Markdown
from sklearn.linear_model import PassiveAggressiveClassifier
linear_clf = PassiveAggressiveClassifier(max_iter = 50)
add Codeadd Markdown
linear_clf.fit(X_train, y_train)
pred = linear_clf.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy: %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['Fake', 'Real'])
add Codeadd Markdown
pred = linear_clf.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
score
