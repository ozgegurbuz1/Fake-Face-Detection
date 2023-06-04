# -*- coding: utf-8 -*-
"""
@author: ozgeg
"""
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Modeli yükle
model = load_model('models/vgg16/fakeFaceDetection.h5')

# Test edilecek görüntüyü yükle
test_image = image.load_img('real1.jpg', target_size=(224, 224))
plt.imshow(test_image)
plt.show()

# Görüntüyü modele uygun hale getir
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# Sınıflandırma yap ve sonucu yazdır
result = model.predict(test_image)
print(result)

# Sınıf etiketlerini belirleme.flow_from_directory fonksiyonu tarafından otomatik olarak sınıf etiketleri atanır. Bu durumda etiketler "fake" sınıfı için 0, "real" sınıfı için 1 indeksine karşılık gelecektir. 
label_mapping = {0: 'fake', 1: 'real'}
predicted_class_index = np.argmax(result)
predicted_class_label = label_mapping[predicted_class_index]
print(predicted_class_label)
