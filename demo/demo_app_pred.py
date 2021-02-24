import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import signal
import time
import numpy as np

from demo_ai import get_predictive_ai as get_ai, train_ai, load_ai, map

from flask import jsonify
from web.app import get_app
app = get_app()

BASE_DIR = os.path.dirname(__file__)

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

import random
data = [[[map(25, 22, 28, 0, 1)] for _ in range(2)]]

ai = get_ai()
ai.summary()

d = np.asarray([[data[0][0]]]).astype('float32')
h = np.zeros(shape=(1, 1))
c = np.zeros(shape=(1, 1))

x, h, c = ai.predict([ np.asarray(data).astype('float32'), h, c])

print("x: ", x)
print("c: ", c)
print()
x, h, c = ai.predict([ np.asarray(data).astype('float32'), h, c])
# x, h, c = ai.predict([ d, h, c])
print("new x:", x)
print("new c:", c)
print()

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