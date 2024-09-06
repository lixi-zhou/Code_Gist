import tensorflow as tf
import h5py

nn = tf.keras.models.Sequential([
    tf.keras.Input(shape=(3)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])
nn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

nn_weights = nn.get_weights()

with h5py.File('./llm_mr_ffnn/model.h5', 'w') as f:
    f.create_dataset('w1', data=nn_weights[0])
    f.create_dataset('b1', data=nn_weights[1])
    f.create_dataset('w2', data=nn_weights[2])
    f.create_dataset('b2', data=nn_weights[3])