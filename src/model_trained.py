import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Path
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
os.chdir(project_root)

TRAIN_DIR = r"train_data"
TEST_DIR  = r"test_data"
SAMPLE_RATE = 16000

# Load YAMNet
yamnet_model_handle = "yamnet_tensorflow"
yamnet_model = hub.load(yamnet_model_handle)

# Extract embedding（YAMNet）
def extract_embedding(file_path):
    waveform, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    waveform = waveform[:SAMPLE_RATE * 10]  # first 10 second
    waveform_tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)
    scores, embeddings, spectrogram = yamnet_model(waveform_tensor)
    embedding = tf.reduce_mean(embeddings, axis=0).numpy()
    return embedding

# Extract lable
def extract_label(filename):
    return filename.split('_')[0]  # dog_1.wav -> dog

# Load dataset
def load_dataset(audio_dir):
    X, y = [], []
    for filename in os.listdir(audio_dir):
        if filename.endswith('.wav'):
            path = os.path.join(audio_dir, filename)
            label = extract_label(filename)
            try:
                embedding = extract_embedding(path)
                X.append(embedding)
                y.append(label)
            except Exception as e:
                print(f"load failed: {filename} → {e}")
    return np.array(X), np.array(y)

# Train classifier
def train_classifier(X, y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y_encoded)
    return clf, le

# Predict single file
def predict_audio(file_path, clf, le):
    embedding = extract_embedding(file_path)
    pred = clf.predict([embedding])[0]
    prob = clf.predict_proba([embedding])[0][pred]
    label = le.inverse_transform([pred])[0]
    print(f" {os.path.basename(file_path)} → predict: {label}, confidence: {prob:.2f}")
    return label, prob

# main
if __name__ == '__main__':
    print("Load training data...")
    X_train, y_train = load_dataset(TRAIN_DIR)

    print("Train classifier...")
    classifier, label_encoder = train_classifier(X_train, y_train)

    print("\nPredicting test dataset:")
    for filename in os.listdir(TEST_DIR):
        if filename.endswith('.wav'):
            test_path = os.path.join(TEST_DIR, filename)
            predict_audio(test_path, classifier, label_encoder)

    print("\nAll done")

    
# Save trained model
#print("\nSaving model")

#import pickle

# create model file
#os.makedirs('model', exist_ok=True)

# save classifier和label_encoder to pkl file
#with open('model/yamnet_model.pkl', 'wb') as f:
#    pickle.dump((classifier, label_encoder), f)

#print("model saved to: model/yamnet_model.pkl")
