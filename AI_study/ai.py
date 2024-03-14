from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

text=['나는 기분이 좋아','나는 짜증나','존나 화나네','기모찌']
labels=[1,0,0,1]

text_train, text_test, y_train, y_test=train_test_split(text, labels, test_size=0.2, random_state=43)

ct=CountVectorizer()
X_train=ct.fit_transform(text_train)
X_test=ct.transform(text_test)

model=MultinomialNB()
model.fit(X_train,y_train)

predictions=model.predict(X_test)
print("정확도",accuracy_score(y_test,predictions))
