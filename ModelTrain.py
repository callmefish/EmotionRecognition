import argparse
import numpy as np
import tensorflow as tf
from models import ACRNN
import pickle
import os
import datetime
import io
import matplotlib.pyplot as plt
import sklearn.metrics
import itertools
import shutil

# hyper parameter of training
parser = argparse.ArgumentParser(description='Speech Emotion Recognition Based on 3D CRNN')
parser.add_argument('--num_epoch', default=100, type=int, help='The number of epoches for training.')
parser.add_argument('--num_classes', default=4, type=int, help='The number of emotion classes.')
parser.add_argument('--batch_size', default=32, type=int, help='The number of samples in each batch.')
parser.add_argument('--learning_rate', default=0.0001, type=float, help='learning rate of Adam optimizer')
parser.add_argument('--dropout_keep_prob', default=0.5, type=float, help='the prob of every unit keep in dropout layer')
parser.add_argument('--traindata_path', default='/content/IEMOCAP_4.pkl', type=str, help='total dataset includes training set')
parser.add_argument('--log_dir', default='/content/logs/gradient_tape/', type=str, help='tensorboard log dir')
parser.add_argument('--checkpoint', default='/content/save/', type=str, help='the checkpoint dir')
parser.add_argument('--model_name', default='model_IEMOCAP_4.ckpt', type=str, help='model name')

arg = parser.parse_args()


class model_training():
    def load_data(self, in_dir):
        f = open(in_dir, 'rb')
        Train_data, Train_label, Val_data, Val_label, Test_data, Test_label = pickle.load(f)
        return Train_data, Train_label, Val_data, Val_label, Test_data, Test_label

    def plot_to_image(self, figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

    def plot_confusion_matrix(self, cm, class_names):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.

        Args:
            cm (array, shape = [n, n]): a confusion matrix of integer classes
            class_names (array, shape = [n]): String names of the integer classes
        """
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Compute the labels from the normalized confusion matrix.
        labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return figure

    def log_confusion_matrix(self, mymodel, testing_data, testing_label, file_writer_cm, epoch):
        # Use the model to predict the values from the validation dataset.
        test_pred_raw = mymodel.predict(testing_data)
        test_pred = np.argmax(test_pred_raw, axis=1)

        # Calculate the confusion matrix.
        cm = sklearn.metrics.confusion_matrix(testing_label, test_pred)
        # Log the confusion matrix as an image summary.
        figure = self.plot_confusion_matrix(cm, class_names=class_names)
        cm_image = self.plot_to_image(figure)

        # Log the confusion matrix as an image summary.
        with file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)

    @tf.function
    def train_step(self, model, data, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = model(data, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(self, model, data, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(data, training=False)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    @tf.function
    def valid_step(self, model, data, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(data, training=False)
        t_loss = loss_object(labels, predictions)

        valid_loss(t_loss)
        valid_accuracy(labels, predictions)
    

if __name__ == '__main__':
    modelTrain = model_training()
    # load data
    train_data, train_label, valid_data, valid_label, test_data, test_label = modelTrain.load_data(arg.traindata_path)
    class_names = ["hap", "ang", "sad", "neu"]

    train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_label)).shuffle(10000).batch(arg.batch_size)
    valid_ds = tf.data.Dataset.from_tensor_slices((valid_data, valid_label)).batch(arg.batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((test_data, test_label)).batch(arg.batch_size)

    # build model
    model = ACRNN(num_classes=arg.num_classes,
                    dropout_keep_prob=arg.dropout_keep_prob)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=arg.learning_rate)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = arg.log_dir + current_time + '/train'
    valid_log_dir = arg.log_dir + current_time + '/valid'
    cm_log_dir = arg.log_dir + current_time + '/cm'
    plot_log_dir = arg.log_dir + current_time + '/plot'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)
    file_writer_cm = tf.summary.create_file_writer(cm_log_dir)


    EPOCHS = arg.num_epoch
    pre_acc = 0.

    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()

        for mel_data, labels in train_ds:
            modelTrain.train_step(model, mel_data, labels)

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
        
        if epoch % 10 == 0:
            modelTrain.log_confusion_matrix(model, valid_data, valid_label, file_writer_cm, epoch)
            print("draw confusion matrix")

        for mel_data_valid, valid_labels in valid_ds:
            modelTrain.valid_step(model, mel_data_valid, valid_labels)
        
        with valid_summary_writer.as_default():
            tf.summary.scalar('loss', valid_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', valid_accuracy.result(), step=epoch)
        
        if valid_accuracy.result() > pre_acc:
            model.save_weights(arg.checkpoint)
            print("model saved to {} the acc is {} and the loss is {}".format(arg.checkpoint, 
                                                                    valid_accuracy.result(),
                                                                    valid_loss.result()))
            pre_acc = valid_accuracy.result()

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Valid Loss: {valid_loss.result()}, '
            f'Valid Accuracy: {valid_accuracy.result() * 100}'
        )