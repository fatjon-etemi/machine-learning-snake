import matplotlib.pyplot as plt
import numpy as np

selected_dataset = 'saved_expert_player.npy'


def plot_training(history):
    # summarize history for accuracy
    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='lower right')
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper right')
    plt.show()


def plot_data():
    dataset = np.load(selected_dataset, allow_pickle=True)[0]
    plt.figure(figsize=(4, 4))
    X = np.array([i[0] for i in dataset])
    X = X.reshape(-1, 24)
    y = np.array([np.argmax(i[1]) for i in dataset])
    plt.plot(X, y, 'ro')
    plt.show()


if __name__ == '__main__':
    plot_data()
