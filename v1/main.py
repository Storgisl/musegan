import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print("Memory growth setting failed:", e)

from musegan.core import MuseGAN
from musegan.components import TemporalHybrid
from input_data import InputDataTemporalHybrid
from config import TrainingConfig, TemporalHybridConfig
import SharedArray as sa

def ensure_shared_array_exists(name, shape=(100, 96, 84, 5), dtype=np.float32):
    try:
        sa.attach(name)
        print(f"[✓] SharedArray '{name}' already exists.")
    except FileNotFoundError:
        print(f"[!] SharedArray '{name}' not found. Creating dummy data...")
        shared_arr = sa.create(name, shape, dtype)
        shared_arr[:] = np.random.rand(*shape).astype(dtype)
        print(f"[+] Created SharedArray '{name}' with dummy data.")

# === Config and Model
t_config = TrainingConfig()
t_config.exp_name = 'exps/temporal_hybrid'
path_x_train_phr = 'tra_X_phrase_all'

# Ensure the SharedArray exists
ensure_shared_array_exists(path_x_train_phr)

model = TemporalHybrid(TemporalHybridConfig())
input_data = InputDataTemporalHybrid(model)
input_data.add_data_sa(path_x_train_phr, 'train')

# === Create and Train MuseGAN
with tf.Session(config=tf.ConfigProto()) as sess:
    musegan = MuseGAN(sess, t_config, model)
    musegan.train(input_data)
#   musegan.generate(input_data)
#    musegan.load_weights(musegan.dir_ckpt)
#    musegan.gen_test(input_data, is_eval=True)

