#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout


# In[2]:


(x_train, y_train), (x_test, y_test) = mnist.load_data("/Users/alireza/Desktop/CI/mnist.npz")
x_train = np.array(x_train).reshape(60000,784)
x_test = np.array(x_test).reshape(10000,784)
print(np.shape(x_train))
print(np.shape(x_test))
x_train = keras.utils.normalize(x_train)
x_test = keras.utils.normalize(x_test)


# In[3]:


model = Sequential()
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
sgd = keras.optimizers.sgd(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=10)


# In[4]:


print(model.history.history.keys())


# In[5]:


fig, ax = plt.subplots()
x = range(1,len(model.history.history['loss'])+1)
y = range(0,101,10)
ax.grid(color='gray', alpha=0.25)
ax.set_axisbelow(True)
plt.title("Test Data")
plt.xlabel("Epoch")
plt.ylabel("Rate(%)")
plt.xticks(x)
plt.yticks(y)
plt.ylim(-5,105)
ax.plot(x,100*np.array(model.history.history['val_loss']), label= 'loss')
ax.plot(x,100*np.array(model.history.history['val_accuracy']), label='accuracy')
ax.legend()
plt.savefig("Q4-test.png")
plt.show()

fig, ax = plt.subplots()
ax.grid(color='gray', alpha=0.25)
ax.set_axisbelow(True)
plt.title("Training Data")
plt.xlabel("Epoch")
plt.ylabel("Rate(%)")
plt.xticks(x)
plt.yticks(y)
plt.ylim(-5,105)
ax.plot(x,100*np.array(model.history.history['loss']), label= 'loss')
ax.plot(x,100*np.array(model.history.history['accuracy']), label='accuracy')
ax.legend()
plt.savefig("Q4-train.png")
plt.show()


# In[6]:


test_loss, test_acc = model.evaluate(x_test, y_test)
test_acc


# In[7]:


model = Sequential()
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(10, activation='softmax'))
sgd = keras.optimizers.sgd(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=10)


# In[8]:


fig, ax = plt.subplots()
x = range(1,len(model.history.history['loss'])+1)
y = range(0,101,10)
ax.grid(color='gray', alpha=0.25)
ax.set_axisbelow(True)
plt.title("Test Data")
plt.xlabel("Epoch")
plt.ylabel("Rate(%)")
plt.xticks(x)
plt.yticks(y)
plt.ylim(-5,105)
ax.plot(x,100*np.array(model.history.history['val_loss']), label= 'loss')
ax.plot(x,100*np.array(model.history.history['val_accuracy']), label='accuracy')
ax.legend()
plt.savefig("Q4-test-dropout.png")
plt.show()

fig, ax = plt.subplots()
ax.grid(color='gray', alpha=0.25)
ax.set_axisbelow(True)
plt.title("Training Data")
plt.xlabel("Epoch")
plt.ylabel("Rate(%)")
plt.xticks(x)
plt.yticks(y)
plt.ylim(-5,105)
ax.plot(x,100*np.array(model.history.history['loss']), label= 'loss')
ax.plot(x,100*np.array(model.history.history['accuracy']), label='accuracy')
ax.legend()
plt.savefig("Q4-train-dropout.png")
plt.show()


# In[9]:


test_loss, test_acc = model.evaluate(x_test, y_test)
test_acc


# In[ ]:




