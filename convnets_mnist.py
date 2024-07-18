#!/usr/bin/env python
# coding: utf-8

# In[7]:


from keras import layers
from keras import models


# In[9]:


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))


# In[10]:


model.summary()


# In[11]:


model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


# In[12]:


model.summary()


# In[13]:


from keras.datasets import mnist
from keras.utils import to_categorical


# In[14]:


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# In[15]:


train_images = train_images.reshape(60000, 28, 28, 1)
train_images = train_images.astype('float32')/255


# In[17]:


test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32')/255


# In[19]:


train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# In[20]:


model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[22]:


model.fit(train_images, train_labels, epochs = 5, batch_size = 64)


# In[23]:


test_loss, test_acc = model.evaluate(test_images, test_labels)


# In[24]:


test_acc


# In[ ]:




