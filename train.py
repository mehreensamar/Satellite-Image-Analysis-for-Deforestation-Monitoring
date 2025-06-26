from preprocess import load_dataset
from model import build_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from config import IMAGE_SIZE
import numpy as np

X, y = load_dataset("data/processed/")
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

model = build_model((128, 128, 3), len(set(y)))
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20)
model.save("deforestation_cnn_model.h5")
