import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import signal
import time
import numpy as np
import itertools
import matplotlib.pyplot as plt

from demo_ai import get_predictive_ai as get_ai, train_ai, load_ai, map

from flask import jsonify
from web.app import get_app
import math
app = get_app()

BASE_DIR = os.path.dirname(__file__)
import tensorflow as tf
from tensorflow import keras

from threading import Thread

AI_MIN_OK = 22
AI_MAX_OK = 28
AI_LOC = 25
AI_SCL = 2.2

def exit_handler():
    print(f'Stopping server.')
    sys.exit()

AI_PRE_WEIGHTS_PATH = os.path.join(BASE_DIR, "ai_data/weights_prel.h5")
AI_ACT_WEIGHTS_PATH = os.path.join(BASE_DIR, "ai_data/weights.h5")

# ai_model = get_ai()
# if os.path.exists(AI_ACT_WEIGHTS_PATH):
#     print(f'Found previous weights, loading weights...')
#     load_ai(ai_model, AI_ACT_WEIGHTS_PATH)
# else:
#     print(f'Found no previous weights, retraining model...')
#     try:
#         train_ai(ai_model, AI_MIN_OK, AI_MAX_OK, AI_LOC, AI_SCL, batch_size=32, epochs=25)
#         ai_model.save_weights(AI_PRE_WEIGHTS_PATH)
#         print(f'Saved weights at "{AI_PRE_WEIGHTS_PATH}". Remember to rename to "{AI_ACT_WEIGHTS_PATH}" to load them for next start.')
#     except KeyboardInterrupt:
#         print('Received keyboard interrupt')
#         exit_handler()
#     except Exception as e:
#         print(str(e))

@app.route('/api/predict/<value>')
def api_predict(value):
    try:
        value = float(value)
        mapped = map(value, AI_MIN_OK, AI_MAX_OK, 0, 1)

        pred = ai_model.predict([[mapped]])[0]

        ok = pred[0] > pred[1]
        confidence = max(pred[0], pred[1])

        result = "ok" if ok else "not ok"
        return jsonify({ 
            "status": "ok", 
            "result": result, 
            "confidence": f'{(confidence * 100):2.0f}%'
        })

    except Exception as e:
        print(str(e))
        return jsonify({ "status": "error", "message": "Internal server error" })

def convert_to_seq_data(data, min_val, max_val, seq_length=4):
    X = []
    Y = []
    sequences = len(data) - seq_length - 1
    for i in range(sequences):
        sequence = data[i:i+seq_length+1]
        X.append([[i] for i in sequence[:-1]])
        Y.append([[i] for i in sequence[1:]])

    return np.asarray(X).astype('float32'), np.asarray(Y).astype('float32')

import random
seq_length = 32
state_size = 64
BATCH_SIZE = 128
EPOCHS = 10
normal_distrib = np.random.normal(0.5, 0.2, size=10000)
X, Y = convert_to_seq_data(normal_distrib, 0.1, 0.9, seq_length=seq_length)

ai = get_ai(state_size)

loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam()

prev_loss = 0
print('Initializing training...')
for epoch in range(EPOCHS):
    epoch_loss = 0
    BATCHES = math.ceil(len(X) / BATCH_SIZE)
    for i in range(BATCHES):
        batch_x, batch_y = X[i * BATCH_SIZE:(i+1)*BATCH_SIZE], Y[i * BATCH_SIZE:(i+1)*BATCH_SIZE]
        batch_y = [[y[-1][0]] for y in batch_y]
        loss_total = 0
        with tf.GradientTape() as tape:
            h = np.zeros(shape=(len(batch_x), state_size))
            c = np.zeros(shape=(len(batch_x), state_size))
            logits, _, _ = ai([batch_x, h, c], training=True)
            loss_value = loss_fn(batch_y, logits)
            loss_total += float(loss_value)
        grads = tape.gradient(loss_value, ai.trainable_weights)
        optimizer.apply_gradients(zip(grads, ai.trainable_weights))
        if i == 0:
            print(loss_total)

    # for i, (x, y) in enumerate(zip(X, Y)):
    #     with tf.GradientTape() as tape:
    #         h = np.zeros(shape=(seq_length, state_size))
    #         c = np.zeros(shape=(seq_length, state_size))

    #         logits, _, _ = ai([x, h, c], training=True)
    #         loss_value = loss_fn(y, logits)
    #         loss_total += float(loss_value)

    #         print(x.shape, logits.shape)
    #         exit()

    #         if i % 500 == 0:
    #             print('\nPred: ', ' '.join([f'{float(j[0]):.4f}' for j in logits]))
    #             print('True: ', ' '.join([f'{j[0]:.4f}' for j in y]))
    #             print('Loss: ', float(loss_value), '-' if float(loss_value) < prev_loss else '+')
    #             print()
    #             prev_loss = float(loss_value)
    #     grads = tape.gradient(loss_value, ai.trainable_weights)
    #     optimizer.apply_gradients(zip(grads, ai.trainable_weights))

        # batch_num = batch // BATCH_SIZE
        # if batch // BATCH_SIZE % 10 == 0 or batch_num == math.floor(len(X) / BATCH_SIZE) - 1:
        #     print(f'-- BATCH {batch // BATCH_SIZE:3} completed with loss: {loss_total}')
        #     epoch_loss += loss_total
    print(f'## EPOCH {epoch:3} completed with loss: {epoch_loss}')
print('Training complete.')

norm = normal_distrib[:seq_length+2]
X, _ = convert_to_seq_data(norm, 0.1, 0.9, seq_length=seq_length)
X = X[0]
h = np.zeros(shape=(seq_length, state_size))
c = np.zeros(shape=(seq_length, state_size))

print(np.asarray(X).shape)
# print(h.shape)

data = [v[0] for v in X]
X, h, c = ai.predict([X, h, c])

print("Generating sequences")
x, h, c = np.asarray([X[-1]]), np.asarray([h[-1]]), np.asarray([c[-1]])
for _ in range(seq_length):
    # print(x.shape, h.shape, c.shape)
    print(f"Input:  {x} | {h} {c}")
    x, h, c = ai.predict([x, h, c])
    print(f'Output: {x} | {h} {c}\n')
    data.append(x[0][0])

plt.plot(normal_distrib[:seq_length*2])
plt.plot(data)
plt.show()



# server = Thread(target=app.run)
# server.daemon = True
# server.start()
# while server.is_alive():
#     try:
#         time.sleep(1)
#     except KeyboardInterrupt:
#         exit_handler()
#     except:
#         raise