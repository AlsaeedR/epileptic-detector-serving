import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import Sequential

# Ensure TensorFlow is running in eager mode
tf.config.run_functions_eagerly(True)

# -------------------- CONSTANTS --------------------
SEQUENCE_LENGTH = 30
FEATURE_BATCH_SIZE = 4
IMAGE_SIZE = (224, 224)
MODEL_PATH = os.path.join("models", "seizure_lstm_model.h5")

# -------------------- MODEL LOADING --------------------
@st.cache_resource
def load_seizure_model():
    def custom_lstm_layer(**kwargs):
        kwargs.pop('time_major', None)
        return LSTM(**kwargs)

    try:
        model = load_model(
            MODEL_PATH,
            custom_objects={'LSTM': custom_lstm_layer}
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        dummy_input = np.zeros((1, SEQUENCE_LENGTH, 1280), dtype=np.float32)
        model.predict(dummy_input, verbose=0)
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

# -------------------- FEATURE EXTRACTOR --------------------
@st.cache_resource
def load_feature_extractor():
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False
    model = Sequential([
        base_model,
        GlobalAveragePooling2D()
    ])
    return model

# -------------------- VIDEO PROCESSING --------------------
# -------------------- PREDICTION --------------------
def predict_seizure(video_path, model, feature_extractor):
    cap = cv2.VideoCapture(video_path)
    sequence = []
    prediction_sum = None
    sequence_count = 0
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame = cv2.resize(frame, IMAGE_SIZE)
            frame = tf.keras.applications.mobilenet_v2.preprocess_input(frame)
            sequence.append(frame)

            if len(sequence) == SEQUENCE_LENGTH:
                sequence_batch = np.asarray(sequence, dtype=np.float32)
                sequence.clear()
                features = feature_extractor.predict(
                    sequence_batch, batch_size=FEATURE_BATCH_SIZE, verbose=0
                )
                sequence_prediction = model.predict(
                    np.expand_dims(features, axis=0), verbose=0
                )[0]
                if prediction_sum is None:
                    prediction_sum = sequence_prediction.copy()
                else:
                    prediction_sum += sequence_prediction
                sequence_count += 1

        if frame_count < SEQUENCE_LENGTH:
            return "Error: Video too short (needs at least 30 frames)", None, None
        if sequence_count == 0:
            return "Error: Could not create sequences", None, None

        avg_pred = prediction_sum / sequence_count
        class_idx = int(np.argmax(avg_pred))

        # Manual label map
        label_map = {0: 'No_Seizure', 1: 'P', 2: 'PG'}
        label = label_map.get(class_idx, str(class_idx))
        confidence = float(avg_pred[class_idx])

        return f"Prediction: {label} (Confidence: {confidence:.2%})", label, confidence

    except Exception as e:
        return f"Error during prediction: {str(e)}", None, None
    finally:
        cap.release()

# -------------------- STREAMLIT APP --------------------
def main():
    st.set_page_config(
        page_title="Seizure Detection System",
        layout="wide",
        page_icon="⚕️"
    )

    with st.spinner("Loading resources..."):
        model = load_seizure_model()
        feature_extractor = load_feature_extractor()

    if model is None:
        st.error("Critical Error: Could not load required resources")
        st.stop()

    st.title("⚡ Seizure Detection from Video")
    st.markdown("Upload a video file to analyze for seizure activity.")

    uploaded_file = st.file_uploader(
        "Choose a video file", 
        type=["mp4", "avi", "mov"]
    )

    if uploaded_file:
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        video_path = os.path.join(temp_dir, uploaded_file.name)

        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        col1, col2 = st.columns(2)
        with col1:
            st.video(video_path)

        with col2:
            with st.spinner("Analyzing video..."):
                result, label, confidence = predict_seizure(
                    video_path, model, feature_extractor)
            
            if label:
                st.success(result)
                st.metric("Prediction", label)
                st.metric("Confidence", f"{confidence:.2%}")
            else:
                st.error(result)

        try:
            os.remove(video_path)
        except:
            pass

if __name__ == "__main__":
    main()
