# api.py
import io
import json
import os
import urllib.request
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

MODEL_URL = os.environ.get("MODEL_URL", "")
MODEL_PATH = Path("checkpoints/model.h5")

def download_model_if_needed():
    """Model dosyasını URL'den indir (eğer yoksa)"""
    if MODEL_PATH.exists():
        print(f"Model zaten mevcut: {MODEL_PATH}")
        return
    
    if not MODEL_URL:
        # Local checkpoints'e bak
        ckpts = sorted(Path("checkpoints").glob("*.h5"))
        if ckpts:
            return  # Local checkpoint var, indirmeye gerek yok
        raise ValueError("MODEL_URL environment variable gerekli veya checkpoints/ klasöründe .h5 dosyası olmalı!")
    
    print(f"Model indiriliyor: {MODEL_URL}")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"Model indirildi: {MODEL_PATH}")

def load_latest_checkpoint():
    download_model_if_needed()
    
    if MODEL_PATH.exists():
        ckpt = MODEL_PATH
    else:
        ckpts = sorted(Path("checkpoints").glob("*.h5"))
        if not ckpts:
            raise FileNotFoundError("checkpoints/ klasöründe .h5 dosyası bulunamadı!")
        ckpt = ckpts[-1]
    
    print("Loaded checkpoint:", ckpt)
    model = keras.models.load_model(ckpt)
    
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