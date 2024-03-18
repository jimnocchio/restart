# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# text=['나는 기분이 좋아','나는 짜증나','존나 화나네','기모찌']
# labels=[1,0,0,1]

# text_train, text_test, y_train, y_test=train_test_split(text, labels, test_size=0.2, random_state=43)

# ct=CountVectorizer()
# X_train=ct.fit_transform(text_train)
# X_test=ct.transform(text_test)

# model=MultinomialNB()
# model.fit(X_train,y_train)

# predictions=model.predict(X_test)
# print("정확도",accuracy_score(y_test,predictions))

# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.datasets import load_iris
# from sklearn.metrics import accuracy_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression


# iris=load_iris()
# X=iris.data
# y=iris.target

# X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3, random_state=54)
# model=DecisionTreeClassifier()
# model.fit(X_train, y_train)

# predictions=model.predict(X_test)
# print("정확도",accuracy_score(y_test, predictions))

import cv2
# import matplotlib.pyplot as plt

# image_path='Ai_study/example.jpg'
# image=cv2.imread(image_path)

# blurred_image=cv2.GaussianBlur(image,(5,5),0)

# plt.figure(figsize=(10,5))

# plt.subplot(2,2,1)
# plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
# plt.title('Original Image')

# plt.subplot(2,2,2)
# plt.imshow(cv2.cvtColor(blurred_image,cv2.COLOR_BGR2RGB))
# plt.title('Blurred Image')
# plt.show()

# face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+ 'haarcascade_frontalface_default.xml')
# img=cv2.imread('c:/programmer jim/Ai_study/ex.png')
# gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# faces=face_cascade.detectMultiScale(gray,1.1,4)
# for (x,y,w,h) in faces:
#     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# cv2.imshow('img',img)
# cv2.waitKey()    

import tensorflow as tf
from tensorflow.keras import layers, models

# MNIST 데이터셋 로드
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 데이터 전처리
train_images = train_images / 255.0
test_images = test_images / 255.0

# CNN 모델 구축
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
model.fit(train_images, train_labels, epochs=2)

# 모델 평가
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('테스트 정확도:', test_acc)
