import tensorflow as tf
import tensorflow_datasets as tfds

mnist_data = tfds.load(name="fashion_mnist", with_info=True)
for item in mnist_data:
    print(item)

mnist_train = tfds.load(name="fashion_mnist", split="train")
assert isinstance(mnist_train, tf.data.Dataset)
print(type(mnist_train))

for item in mnist_train.take(1):
    print(type(item))
    print(item.keys())
    print(item['image'])
    print(item['label'])

(training_images, training_labels), (test_images, test_labels) = tfds.as_numpy(
    tfds.load("fashion_mnist", split=["train", "test"], batch_size=-1, as_supervised=True))

training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax)
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(training_images, training_labels, epochs=5)
