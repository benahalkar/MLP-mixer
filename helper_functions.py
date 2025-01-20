import matplotlib.pyplot as plt
import os
import shutil
import numpy as np

# Function to create and save plots
def save_plots(file_path, history, model_name):
  # plot and save training and validation accuracies  
  plt.plot(history.history['accuracy'], label='Train Accuracy')
  plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
  plt.title('Model Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.tight_layout()
  plt.savefig(os.path.join(file_path, model_name, 'accuracy_plot.png'))
  plt.show()
  
  # plot and save training and validation losses
  plt.plot(history.history['loss'], label='Train Loss')
  plt.plot(history.history['val_loss'], label='Validation Loss')
  plt.title('Model Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.tight_layout()
  plt.savefig(os.path.join(file_path, model_name, 'loss_plot.png'))
  plt.show()


# Function to save models
def save_model(file_path, model, model_name):
  # save model in .h5 format
  model.save(os.path.join(file_path, model_name, "model.h5"))


# Function to save accuracy values
def save_accuracy(file_path, history, model_name):
    # save accuracy values in .npy format
    np.save(os.path.join(file_path, model_name, 'accuracy_values.npy'), history.history)


# Function for weight visualization
def visualize_weights(file_path, model, model_name):
    # save plot of weight visualization of all layers
    if not os.path.exists(os.path.join(file_path, model_name, "weights")):
        os.makedirs(os.path.join(file_path, model_name, "weights"))

    for layer in model.layers:
        if 'conv' in layer.name:
            layer_name = layer.name
            weights = layer.get_weights()[0]

            if len(weights.shape) == 4:
                for i in range(weights.shape[3]):
                    plt.imshow(weights[:, :, 0, i], cmap='viridis', interpolation='none')
                    plt.colorbar()
                    plt.title("{}_Channel_{}".format(layer_name, i+1))
                    plt.savefig(os.path.join(file_path, model_name, "weights", "{}_Channel_{}.png".format(layer_name, i+1)))
                    plt.close()
            else:
                print("Visualization not supported for {}".format(layer_name))


# main function to save all data
def save_data(file_path, model, history, model_name):
    # wrapper function to perform all above tasks
    if os.path.exists(os.path.join(file_path, model_name)):
      shutil.rmtree(os.path.join(file_path, model_name))
    
    os.makedirs(os.path.join(file_path, model_name))

    save_plots(file_path, history, model_name)
    save_model(file_path, model, model_name)
    save_accuracy(file_path, history, model_name)
    visualize_weights(file_path, model, model_name)

