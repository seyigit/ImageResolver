# api.py
import io
import json
from typing import Dict

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow import keras
from pathlib import Path

# TensorFlow optimizasyonları
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

def load_latest_checkpoint():
    ckpts = sorted(Path("checkpoints").glob("*.h5"))
    if not ckpts:
        raise FileNotFoundError("checkpoints/ klasöründe .h5 dosyası bulunamadı!")
    print("Loaded checkpoint:", ckpts[-1])
    model = keras.models.load_model(ckpts[-1])
    
    # İlk predict'i başlangıçta yap (warmup)
    dummy_input = np.zeros((1, 60, 150, 1), dtype=np.float32)
    model.predict(dummy_input, verbose=0)
    print("Model warmup tamamlandı")
    
    return model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model + config yükle
model = load_latest_checkpoint()

with open("ocr6_config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)

ALPHABET = cfg["alphabet"]
CAPTCHA_LEN = cfg["captcha_len"]
IMG_HEIGHT = cfg["img_height"]
IMG_WIDTH = cfg["img_width"]
CHANNELS = cfg["channels"]

idx_to_char = {i: ch for i, ch in enumerate(ALPHABET)}

def preprocess_tf_bytes(image_bytes: bytes) -> np.ndarray:
    """train_model.py ile birebir aynı TF pipeline, ama bytes üzerinden."""
    # PNG/JPEG otomatik algılama
    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False, dtype=tf.uint8)
    img = tf.image.rgb_to_grayscale(img)  # Grayscale'e çevir
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.cast(img, tf.float32) / 255.0  # [0,1]
    # Kontrast artır
    img = tf.image.adjust_contrast(img, 2.0)
    img = tf.clip_by_value(img, 0.0, 1.0)
    img = tf.expand_dims(img, axis=0)  # (1, H, W, 1)
    return img.numpy()

@tf.function(reduce_retracing=True)
def predict_fast(x):
    """TensorFlow graph modunda çalışır - daha hızlı"""
    return model(x, training=False)

def decode_prediction(pred: np.ndarray) -> str:
    # pred: shape (1, 6, num_classes)
    pred_step = pred[0]  # (6, num_classes)
    indices = np.argmax(pred_step, axis=-1)
    return "".join(idx_to_char[int(i)] for i in indices)

@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, str]:
    contents = await file.read()
    x = preprocess_tf_bytes(contents)
    
    # TensorFlow graph modunda çalıştır
    x_tensor = tf.constant(x, dtype=tf.float32)
    pred = predict_fast(x_tensor).numpy()
    
    text = decode_prediction(pred)
    return {"text": text}