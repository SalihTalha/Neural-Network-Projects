import tensorflow as tf
from os import path, getcwd, chdir

path = f"{getcwd()}/../tmp2/mnist.npz"


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

def train_mnist_conv():

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, log = {}):
            if(log.get('acc') > 0.998):
                self.model.stop_training = True
                print("Reached 99.8% accuracy so cancelling training!")

    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data(path=path)
    callback = myCallback()
    
    training_images = training_images / 255.
    test_images = test_images / 255.
    

    model = tf.keras.models.Sequential([
            # YOUR CODE STARTS HERE
        tf.keras.layers.Reshape([28, 28, 1]),
        tf.keras.layers.Conv2D(64, (3,3), input_shape = (28,28)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(input_shape=(13, 13)),
        tf.keras.layers.Dense(256, activation = tf.nn.relu),
        tf.keras.layers.Dense(10, activation = tf.nn.softmax)

    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model fitting
    history = model.fit(
        x=training_images, y=training_labels, epochs=20, callbacks = [callback]
    )
    return history.epoch, history.history['acc'][-1]

# _, _ = train_mnist_conv()

