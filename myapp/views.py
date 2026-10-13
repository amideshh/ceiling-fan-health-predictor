
import os
import math
import tempfile
import joblib
import librosa
import numpy as np

from django.shortcuts import render
from django.views.decorators.http import require_http_methods


# =====================================================
# PROJECT PATHS
# =====================================================

BASE_DIR = r"D:\AI_Device_Healt_Prediction"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "fan_random_forest.pkl",
)


# =====================================================
# AUDIO FEATURE EXTRACTION
# IMPORTANT: Must match model training features
# =====================================================

def extract_features(audio_path):
    audio, sr = librosa.load(
        audio_path,
        sr=None,
        mono=True,
    )

    if len(audio) == 0:
        raise ValueError("Audio file is empty.")

    features = []

    # MFCC: 13 means + 13 standard deviations
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13,
    )

    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # RMS
    rms = librosa.feature.rms(y=audio)[0]
    features.extend([np.mean(rms), np.std(rms)])

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr,
    )[0]
    features.extend([np.mean(centroid), np.std(centroid)])

    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sr,
    )[0]
    features.extend([np.mean(bandwidth), np.std(bandwidth)])

    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sr,
    )[0]
    features.extend([np.mean(rolloff), np.std(rolloff)])

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(
        y=audio
    )[0]
    features.extend([np.mean(zcr), np.std(zcr)])

    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(
        y=audio,
        sr=sr,
    )
    features.extend(np.mean(contrast, axis=1))
    features.extend(np.std(contrast, axis=1))

    return np.asarray(
        features,
        dtype=float,
    ).reshape(1, -1)


# =====================================================
# LOAD SAVED MODEL
# =====================================================

def load_audio_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Saved audio model not found: " + MODEL_PATH
        )

    return joblib.load(MODEL_PATH)


# =====================================================
# CONVERT PREDICTION INTO READABLE RESULT
# Assumption: 0 = Normal, 1 = Anomaly
# =====================================================

def format_audio_result(prediction):
    label = str(prediction).strip().lower()

    normal_labels = {
        "0",
        "normal",
        "fan normal",
        "good",
        "healthy",
    }

    risk_labels = {
        "1",
        "anomaly",
        "abnormal",
        "bad",
        "risk",
        "fault",
        "fan risk",
    }

    if label in normal_labels:
        return (
            "Fan Normal",
            "normal",
            "The model classified the fan audio as normal.",
        )

    elif label in risk_labels:
        return (
            "Fan Risk Detected",
            "risk",
            "The model detected an abnormal fan sound.",
        )

    return (
        f"Unknown Model Class: {prediction}",
        "unknown",
        "Check the labels used when training the audio model.",
    )


# =====================================================
# SINGLE PAGE: AUDIO + TEMPERATURE
# =====================================================

@require_http_methods(["GET", "POST"])
def predict_fan(request):

    context = {
        "audio_result": None,
        "audio_status": None,
        "audio_message": None,
        "audio_error": None,
        "temperature": None,
        "status": None,
        "error": None,
    }

    if request.method == "POST":

        form_type = request.POST.get(
            "form_type",
            "",
        )

        # =============================================
        # TEMPERATURE CHECKER
        # These are demo thresholds, not a trained model
        # =============================================

        if form_type == "temperature":

            value = request.POST.get(
                "temperature",
                "",
            ).strip()

            try:
                if not value:
                    raise ValueError

                temperature = float(value)

                if not math.isfinite(temperature):
                    raise ValueError

                context["temperature"] = temperature

                if temperature < 25:
                    context["status"] = "Normal"

                elif temperature < 27:
                    context["status"] = "Warning"

                else:
                    context["status"] = "High"

            except (ValueError, TypeError):
                context["error"] = (
                    "Please enter a valid temperature."
                )

            return render(
                request,
                "predict.html",
                context,
            )

        # =============================================
        # FAN AUDIO ANALYSIS
        # =============================================

        if form_type == "audio":

            audio_file = request.FILES.get("audio_file")

            if not audio_file:
                context["audio_error"] = (
                    "Please select a WAV audio file."
                )

            elif not audio_file.name.lower().endswith(".wav"):
                context["audio_error"] = (
                    "Please upload a WAV audio file."
                )

            else:
                temp_path = None

                try:
                    # Save upload temporarily
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".wav",
                    ) as temp_file:

                        for chunk in audio_file.chunks():
                            temp_file.write(chunk)

                        temp_path = temp_file.name

                    # Extract audio features
                    features = extract_features(temp_path)

                    # Load model and predict
                    model = load_audio_model()

                    prediction = model.predict(features)[0]

                    # Convert numpy scalar to normal Python value
                    if hasattr(prediction, "item"):
                        prediction = prediction.item()

                    # Human-readable result
                    (
                        context["audio_result"],
                        context["audio_status"],
                        context["audio_message"],
                    ) = format_audio_result(prediction)

                except Exception as exc:
                    print("Audio prediction error:", exc)

                    context["audio_error"] = (
                        "Audio analysis failed. "
                        "Check the model file, audio file, "
                        "and feature extraction settings."
                    )

                finally:
                    if (
                        temp_path
                        and os.path.exists(temp_path)
                    ):
                        os.remove(temp_path)

    return render(
        request,
        "predict.html",
        context,
    )
