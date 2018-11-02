from PIL import Image
def register_extension(id, extension): Image.EXTENSION[extension.lower()] = id.upper()
Image.register_extension = register_extension
def register_extensions(id, extensions): 
  for extension in extensions: register_extension(id, extension)
Image.register_extensions = register_extensions

import matplotlib.pyplot as plt

def plot_training_loss(history):
  ''' Plot training loss vs. epochs '''

  loss = history['loss']
  epochs = range(len(loss)) 

  plt.figure(figsize=(8, 8))
  plt.plot(epochs, loss, 'bo', label='Training loss')
  plt.title('Training loss')
  plt.xlabel('epochs')
  plt.legend(loc='best')
  plt.grid()

def plot_loss_accuracy(history):
  acc = history['acc']
  val_acc = history['val_acc']
  loss = history['loss']
  val_loss = history['val_loss']

  epochs = range(len(acc))

  # Plotting Accuracy vs Epoch curve
  plt.figure(figsize=(8, 8))

  plt.plot(epochs, acc, 'bo', label='Training acc')
  plt.plot(epochs, val_acc, 'b', label='Validation acc')
  plt.title('Training and validation accuracy')
  #plt.legend(loc='upper left')
  plt.legend(loc='best')
  plt.grid()


  # Plotting Loss vs Epoch curve
  plt.figure(figsize=(8, 8))

  plt.plot(epochs, loss, 'bo', label='Training loss')
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title('Training and validation loss')
  #plt.legend()
  plt.legend(loc='best')
  plt.grid()

def plot_loss_and_metrics(history, metric_name):
  metric = history[metric_name]
  val_metric = history['val_' + metric_name]

  loss = history['loss']
  val_loss = history['val_loss']

  epochs = range(len(metric))

  # Plotting Accuracy vs Epoch curve
  plt.figure(figsize=(8, 8))

  plt.plot(epochs, metric, 'bo', label='Training acc')
  plt.plot(epochs, val_metric, 'b', label='Validation acc')
  plt.title('Training and validation accuracy')
  #plt.legend(loc='upper left')
  plt.legend(loc='best')
  plt.grid()


  # Plotting Loss vs Epoch curve
  plt.figure(figsize=(8, 8))

  plt.plot(epochs, loss, 'bo', label='Training loss')
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title('Training and validation loss')
  #plt.legend()
  plt.legend(loc='best')
  plt.grid()
