from tensorflow.keras.models import load_model
from preprocess import load_dataset
from sklearn.metrics import classification_report

model = load_model("deforestation_cnn_model.h5")
X_test, y_test = load_dataset("data/test/")
y_pred = model.predict(X_test).argmax(axis=1)

print(classification_report(y_test, y_pred))
