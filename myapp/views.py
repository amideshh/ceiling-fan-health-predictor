
import os
import math
import tempfile
import traceback
import joblib
import librosa
import numpy as np

from django.conf import settings
from django.shortcuts import render
from django.views.decorators.http import require_http_methods


# Model path
MODEL_PATH = os.path.join(
    settings.BASE_DIR,
    "fan_random_forest.pkl"
)


def extract_features(audio_path):
    """Extract the 50 audio features used during model training."""

    audio, sr = librosa.load(
        audio_path,
        sr=16000,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError("Audio file is empty.")

    if len(audio) < 2048:
        audio = np.pad(audio, (0, 2048 - len(audio)))

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    rms = librosa.feature.rms(y=audio)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)

    features = []
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))
    features.extend([np.mean(rms), np.std(rms)])
    features.extend([np.mean(centroid), np.std(centroid)])
    features.extend([np.mean(bandwidth), np.std(bandwidth)])
    features.extend([np.mean(rolloff), np.std(rolloff)])
    features.extend([np.mean(zcr), np.std(zcr)])
    features.extend(np.mean(contrast, axis=1))
    features.extend(np.std(contrast, axis=1))

    features = np.asarray(features, dtype=np.float64)

    if len(features) != 50:
        raise ValueError(
            "Expected 50 features, got {}".format(len(features))
        )

    return features.reshape(1, -1)


def load_audio_model():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            "Model file not found at: {}".format(MODEL_PATH)
        )

    return joblib.load(MODEL_PATH)


def format_audio_result(prediction):
    label = str(prediction).strip().lower()

    if label in ("0", "normal", "healthy"):
        return (
            "Fan Normal",
            "normal",
            "The model classified this recording as normal. "
            "This does not guarantee that the fan is safe."
        )

    if label in ("1", "anomaly", "abnormal", "fault"):
        return (
            "Fan Risk Detected",
            "risk",
            "The model detected an unusual sound. "
            "Inspect the fan; this is not a confirmed diagnosis."
        )

    return (
        "Unknown Model Class",
        "unknown",
        "Unexpected prediction label: {}".format(prediction)
    )


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
        form_type = request.POST.get("form_type", "").strip()

        # Temperature checker (demo thresholds, not trained AI)
        if form_type == "temperature":
            value = request.POST.get("temperature", "").strip()

            try:
                if not value:
                    raise ValueError("Temperature is empty.")

                temperature = float(value)

                if not math.isfinite(temperature):
                    raise ValueError("Temperature is not finite.")

                context["temperature"] = temperature

                if temperature < 25:
                    context["status"] = "Normal"
                elif temperature < 27:
                    context["status"] = "Warning"
                else:
                    context["status"] = "High"

            except (ValueError, TypeError):
                context["error"] = "Please enter a valid temperature."

            return render(request, "predict.html", context)

        # Audio prediction
        if form_type == "audio":
            audio_file = request.FILES.get("audio_file")

            if not audio_file:
                context["audio_error"] = "Please select a WAV file."

            elif not audio_file.name.lower().endswith(".wav"):
                context["audio_error"] = "Only WAV files are supported."

            else:
                temp_path = None

                try:
                    with tempfile.NamedTemporaryFile(
                        suffix=".wav",
                        delete=False
                    ) as temp_file:
                        for chunk in audio_file.chunks():
                            temp_file.write(chunk)

                        temp_path = temp_file.name

                    print("Audio file saved:", temp_path)
                    print("Model path:", MODEL_PATH)
                    print("Model exists:", os.path.isfile(MODEL_PATH))

                    features = extract_features(temp_path)
                    print("Feature shape:", features.shape)

                    model = load_audio_model()
                    prediction = model.predict(features)[0]

                    if hasattr(prediction, "item"):
                        prediction = prediction.item()

                    print("Prediction:", repr(prediction))

                    (
                        context["audio_result"],
                        context["audio_status"],
                        context["audio_message"],
                    ) = format_audio_result(prediction)

                except Exception as exc:
                    print("\n===== AUDIO PREDICTION ERROR =====")
                    print("Error:", repr(exc))
                    traceback.print_exc()
                    print("===== END AUDIO ERROR =====\n")

                    context["audio_error"] = (
                        "Audio analysis failed. Please check the "
                        "Django terminal for details."
                    )

                finally:
                    if temp_path and os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except OSError:
                            pass

    return render(request, "predict.html", context)
