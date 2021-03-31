import tensorflow as tf
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from keras.utils import to_categorical
import random
import matplotlib.tri as tri



# Sine sample dataset
# Sine sample dataset
def create_sine(samples):
   X = np.zeros((samples, 2))
   y = np.zeros(samples, dtype='uint8')

   r = np.linspace(-2*np.pi, 2*np.pi, samples)

   for index, x in enumerate(r):

      y_value = (random.randint(1, 200)-100)/100.0



      X[index] = [r[index], y_value]
      if y_value > np.sin(r[index]):

         y[index] = 1
      else:
         y[index] = 0

   return X, y



class Model:

   def __init__(self):
      self.batch_size = 32
      self.classes = 2
      self.features = 2
      self.learning_rate = 0.01
      self.train_loss_results = []
      self.train_accuracy_results = []

      # Create training data and saving as tf dataset
      self.X_train_OG, self.Y_train_OG = create_sine(samples=1000)
      self.train_dataset = tf.data.Dataset.from_tensor_slices((self.X_train_OG, self.Y_train_OG)).batch(self.batch_size)

      # Create testing data
      self.X_test, self.Y_test = create_sine(samples=1000)

   def regen_dataset(self):
      self.X_train_OG, self.Y_train_OG = create_sine(samples=1000)
      self.train_dataset = tf.data.Dataset.from_tensor_slices((self.X_train_OG, self.Y_train_OG)).batch(self.batch_size)
      self.X_test, self.Y_test = create_sine(samples=1000)
      


   def create_model(self, model):

      self.model = tf.keras.Sequential()

      self.features = model['features']
      self.batch_size = model['batch_size']
      self.classes = model['classes']
      self.learning_rate = model['learning_rate']
      layers = model['layers']
      

      for index, layer in enumerate(layers):
         neurons = layer['neurons']
         activation = layer['activation']


         if activation == "ReLU":
            activation = tf.nn.relu
         elif activation == "Softmax":
            activation = tf.nn.softmax
         elif activation == "Linear":
            activation = tf.nn.linear
         elif activation == "TanH":
            activation = tf.nn.tanh

         print(activation)
         if index == 0:
            self.model.add(tf.keras.layers.Dense(neurons, activation=activation, input_shape=(self.features, )))
         else:
            self.model.add(tf.keras.layers.Dense(neurons, activation=activation))

      self.model.add(tf.keras.layers.Dense(self.classes))


      self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
      self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)


   def reset_model(self):
      self.train_loss_results = []
      self.train_accuracy_results = []
      


   def train_model(self, epoch):
      epoch_loss_avg = tf.keras.metrics.Mean()
      epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

      # Training loop - using batches of 32
      for x, y in self.train_dataset:
         # Optimize the model
         loss_value, grads = self.grad(x, y)
         self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

         # Track progress
         epoch_loss_avg.update_state(loss_value)  # Add current batch loss
         # Compare predicted label to actual label
         # training=True is needed only if there are layers with different
         # behavior during training versus inference (e.g. Dropout).
         epoch_accuracy.update_state(y, self.model(x, training=True))

      # End epoch
      self.train_loss_results.append(epoch_loss_avg.result())
      self.train_accuracy_results.append(epoch_accuracy.result())

      if (epoch % 25 == 0):
         print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                      epoch_loss_avg.result(),
                                                                      epoch_accuracy.result()))

      #self.create_graph()
      logits = self.model(self.X_test, training=False)
      self.prediction = tf.argmax(logits, axis=1, output_type=tf.int32)


   def loss(self, x, y, training):
      y_ = self.model(x, training=training)

      return self.loss_object(y_true=y, y_pred=y_)


   def grad(self, inputs, targets):
      with tf.GradientTape() as tape:
         loss_value = self.loss(inputs, targets, training=True)
      return loss_value, tape.gradient(loss_value, self.model.trainable_variables)


   def create_graph(self):

      logits = self.model(self.X_test, training=False)
      prediction = tf.argmax(logits, axis=1, output_type=tf.int32)

      x_min, x_max = self.X_test[:, 0].min(), self.X_test[:, 0].max()
      y_min, y_max = self.X_test[:, 1].min(), self.X_test[:, 1].max()
      h = 0.01
      xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

      zz = np.argmax(self.model(np.c_[xx.ravel(), yy.ravel()], training=False), axis=1)

      zz = zz.reshape(xx.shape)
      plt.contourf(xx, yy, zz)
      plt.scatter(self.X_test[:, 0], self.X_test[:, 1], c=prediction, s=1, cmap=plt.cm.Spectral)
      plt.savefig("output.jpg")

