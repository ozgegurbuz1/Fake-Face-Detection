# -*- coding: utf-8 -*-
"""
@author: ozgeg
"""
import tensorflow
import matplotlib.pyplot as plt
# import os
# from google.colab import drive
# drive.mount('/content/drive')

# VGG16 ve VGG19, önceden eğitilmiş bir evrişimli sinir ağı (CNN) modelleridir.
conv_base = tensorflow.keras.applications.VGG19(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3)
                  )

# Evrişim katmanlarını gösterilir.
conv_base.summary()

# Hangi katmanların eğitildiğine ve dondurulduğuna karar verilir.
# 'block5_conv1' katmanına kadar olanlar dondurulur.
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

# Boş bir model oluşturuldu.
model = tensorflow.keras.models.Sequential()

# VGG16, evrişim katmanı olarak eklendi.
model.add(conv_base)

# Katmanlar matrislerden vektörlere dönüştürüldü.
model.add(tensorflow.keras.layers.Flatten())

# Sinir katmanımız eklendi.
model.add(tensorflow.keras.layers.Dense(256, activation='relu'))
model.add(tensorflow.keras.layers.Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer=tensorflow.keras.optimizers.RMSprop(learning_rate=1e-5),
              metrics=['acc'])

# Oluşturulan model gösteriliyor.
model.summary()

# Verilerin bulunduğu dizinleri tanımlama.
train_dir = 'veriseti/egitim'
validation_dir = 'veriseti/gecerleme'
test_dir = 'veriseti/test'

# Aşırı uyumunu önlemek için veri artırma (data augmentation) yöntemlerini uygulamamız gerekir.
train_datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(
      rescale=1./255, # piksel değerleri 0-255'den 0-1 arasına getiriliyor.
      rotation_range=40, # istenilen artırma işlemleri yapılabilir.
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest'
      )


train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        )

# Eğitim sürecini doğrulamak için artırılmış görüntülere ihtiyacımız yok.
validation_datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
        )

validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=32,
        )

# Modelin eğitimi.
history = model.fit(
      train_generator,
      steps_per_epoch=84,
      epochs=100,
      validation_data=validation_generator,
      validation_steps=1)


plt.plot(history.history['acc']) #her bir epoch için eğitim doğruluğunu çizer.
plt.plot(history.history['val_acc']) #her bir epoch için doğrulama(geçerleme) doğruluğunu çizer.
plt.title('Model acc') #doğruluk grafiğinin başlığını belirler.
plt.xlabel('Epochs') #x-ekseni etiketini ayarlar.
plt.ylabel('acc') #y-ekseni etiketini ayarlar.
plt.legend(['train','val'],loc='upper left') #grafiğe bir açıklama ekler. 'train', eğitim doğruluğunu temsil eder ve 'val', doğrulama(geçerleme) doğruluğunu temsil eder. Açıklama grafiğin sol üst köşesine yerleştirilir.
plt.show() #grafiği görüntüler.

plt.plot(history.history['loss'])#her bir epoch için eğitim kaybını çizer.
plt.plot(history.history['val_loss']) # her bir epoch için doğrulama kaybını çizer.
plt.title('Model loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend(['train','val'],loc='upper left')
plt.show()


# Eğitilmiş model çalışma dizinine kaydedilir.
model.save('fakeFaceDetection.h5')

# Eğitilmiş modeli test etmek için artırılmış görüntülere ihtiyacımız yok.
test_datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
        )

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        )

# Test sonuçlarının yazdırılır.
test_loss, test_acc = model.evaluate(test_generator)
print('test acc:', test_acc)


from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

#Tahmin,Karmaşıklık Matrisi
Y_pred = model.predict(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
cm=confusion_matrix(validation_generator.classes, y_pred)
labels = ["fake", "real"]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.show()
