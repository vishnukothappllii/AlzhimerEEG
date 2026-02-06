import os
import io
import base64
import numpy as np
import mne
import joblib
from flask import Flask, request, render_template, redirect, flash, url_for
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal

# ----------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"set"}
MODEL_PATH = "classifier 1.pkl"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SECRET_KEY"] = "secret123"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ----------------------------------------------------------
# BASIC UTILITIES
# ----------------------------------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_eeg(path):
    """
    Loads the EEG file using MNE.
    Returns raw object (for plotting) plus data, labels, and fs for processing.
    """
    raw = mne.io.read_raw_eeglab(path, preload=True)
    data = raw.get_data()
    labels = raw.ch_names
    fs = int(raw.info["sfreq"])
    return raw, data, labels, fs


def plot_eeg(raw):
    """
    Generates a simple plot of the EEG data for the UI.
    """
    fig = raw.plot(duration=5, n_channels=10, scalings="auto", show=False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ----------------------------------------------------------
# MATH FUNCTIONS (ALZHEIMER)
# ----------------------------------------------------------

def compute_coherence(main_data, compare_data, sample_freq):
    # Frequencies we want to extract
    freq_bounds = [13, 20]
    chan_coherence = np.zeros((1, 2))
    for row_id in range(np.shape(main_data)[0]):
        f, cxy = signal.coherence(compare_data, main_data[row_id, :], sample_freq)
        tf_idx = np.logical_and(f >= freq_bounds[0], f <= freq_bounds[1])
        mean_cxy = np.mean(cxy)
        if mean_cxy == 0:
             chan_coherence[0, row_id] = 0.0
        else:
             chan_coherence[0, row_id] = np.mean(cxy[tf_idx]) / mean_cxy
    return chan_coherence


def frequency_power(data, sample_freq, time_length):
    freq_bounds = [8, 12]
    n = time_length * sample_freq
    frequencies = np.linspace(0, sample_freq / 2, n // 2 + 1)
    x = np.fft.fft(data, axis=1)
    power_spectrum = np.abs(x[:, 0:(n // 2 + 1)]) ** 2
    tf_num = np.logical_and(frequencies >= freq_bounds[0], frequencies <= freq_bounds[1])
    alpha_power = np.mean(power_spectrum[:, tf_num], axis=1)
    return alpha_power


def categorize_patient(data, chn_labels, sample_freq, learner):
    try:
        fz_idx = chn_labels.index('Fz')
        o1_idx = chn_labels.index('O1')
        o2_idx = chn_labels.index('O2')
    except ValueError as e:
        return 0.0, 0.0, [], []

    fz_data = data[fz_idx, :]
    model_data = data[o1_idx, :]
    model_data = np.vstack((model_data, data[o2_idx, :]))

    time_length = 10
    max_testing = 20
    total_len_samples = np.shape(model_data)[1]
    max_possible_epochs = int(total_len_samples / (time_length * sample_freq))
    actual_epochs = min(max_testing, max_possible_epochs)

    epoch_classification = []
    model_statistics = []

    for idx in range(actual_epochs):
        t1 = (idx * time_length * sample_freq)
        t2 = t1 + time_length * sample_freq
        temp_fz = fz_data[t1:t2]
        temp_model_data = model_data[:, t1:t2]

        marker1 = frequency_power(temp_model_data, sample_freq, time_length)
        marker2 = compute_coherence(temp_model_data, temp_fz, sample_freq)
        model_param = np.hstack((marker1, marker2[0]))
        
        model_statistics.append(model_param)
        y_pred = learner.predict(model_param.reshape(1, -1))
        epoch_classification.append(y_pred[0])

    if len(epoch_classification) == 0:
        return 0.0, 0.0, [], []

    percent_control = (epoch_classification.count(0) / len(epoch_classification)) * 100
    percent_abnormal = (epoch_classification.count(1) / len(epoch_classification)) * 100
    
    model_means = np.mean(model_statistics, axis=0)
    model_std = np.std(model_statistics, axis=0)

    return percent_control, percent_abnormal, model_means, model_std

# ----------------------------------------------------------
# DEPRESSION LOGIC (NEW)
# ----------------------------------------------------------
def calculate_depression_score(features):
    """
    Calculates a depression risk score based on Alpha Asymmetry.
    Logic: Depression is often associated with relatively less left frontal activity (higher alpha) 
    compared to right. 
    Formula: Asymmetry = (Right Alpha - Left Alpha) / (Right Alpha + Left Alpha)
    Using O1/O2 as proxy for hemispheric balance if F3/F4 unavailable.
    """
    alpha_left = features['alpha_O1']
    alpha_right = features['alpha_O2']
    
    # Avoid division by zero
    if (alpha_right + alpha_left) == 0:
        return 0.0

    # Calculate Asymmetry Index
    asymmetry = (alpha_right - alpha_left) / (alpha_right + alpha_left)
    
    # Normalize to a 0-100 scale for "Depression Probability"
    # Assuming a threshold: significantly negative asymmetry might indicate risk
    # This is a heuristic mapping for demonstration
    
    # Map -0.5 to 0.5 range to 0-100% probability
    prob = 50 + (asymmetry * 100 * 2) 
    prob = max(0, min(100, prob)) # Clamp between 0 and 100
    
    return prob


# ----------------------------------------------------------
# UI HELPER
# ----------------------------------------------------------
def extract_single_epoch_features_for_ui(data, labels, fs):
    try:
        fz_idx = labels.index('Fz')
        o1_idx = labels.index('O1')
        o2_idx = labels.index('O2')
    except:
        return [0,0,0,0]

    fz_data = data[fz_idx, :]
    model_data = np.vstack((data[o1_idx, :], data[o2_idx, :]))

    time_length = 10
    t1 = 0
    t2 = time_length * fs
    if model_data.shape[1] < t2:
        return [0,0,0,0]

    temp_fz = fz_data[t1:t2]
    temp_model_data = model_data[:, t1:t2]

    marker1 = frequency_power(temp_model_data, fs, time_length)
    marker2 = compute_coherence(temp_model_data, temp_fz, fs)
    
    return [marker1[0], marker1[1], marker2[0][0], marker2[0][1]]


# ----------------------------------------------------------
# ROUTES
# ----------------------------------------------------------

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        return redirect(url_for('login'))
    return render_template('registration.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return redirect(url_for('eeg_interface'))
    return render_template('login.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route("/upload", methods=["GET", "POST"])
def eeg_interface():
    eeg_image = None
    features = None
    filename = None

    if request.method == "POST":
        action = request.form.get("action")

        if action == "upload":
            file = request.files.get("file")
            if file and file.filename != "":
                filename = secure_filename(file.filename)
                path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(path)
                raw, data, labels, fs = load_eeg(path)
                eeg_image = plot_eeg(raw)
        
        elif action == "extract":
            filename = request.form.get("filename")
            if not filename:
                flash("Error: Filename lost. Please upload the file again.")
                return redirect(url_for("eeg_interface"))
            
            path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.exists(path):
                raw, data, labels, fs = load_eeg(path)
                eeg_image = plot_eeg(raw) 
                feats = extract_single_epoch_features_for_ui(data, labels, fs)
                features = {
                    "alpha_O1": float(feats[0]),
                    "alpha_O2": float(feats[1]),
                    "coherence_O1_Fz": float(feats[2]),
                    "coherence_O2_Fz": float(feats[3]),
                }

    return render_template("eeg.html", 
                           filename=filename, 
                           eeg_image=eeg_image, 
                           features=features)


@app.route("/predict", methods=["POST"])
def predict_result():
    """
    Renders index.html (Alzheimer's Results)
    """
    filename = request.form.get("filename")
    if not filename: return redirect(url_for('eeg_interface'))
    path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(path): return redirect(url_for('eeg_interface'))

    raw, data, labels, fs = load_eeg(path)
    try:
        model = joblib.load(MODEL_PATH)
    except:
        flash("Model not found.")
        return redirect(url_for('eeg_interface'))

    percent_control, percent_abnormal, means, std = categorize_patient(data, labels, fs, model)
    eeg_image = plot_eeg(raw)
    
    features = {
        "alpha_O1": float(means[0]),
        "alpha_O2": float(means[1]),
        "coherence_O1_Fz": float(means[2]),
        "coherence_O2_Fz": float(means[3]),
    }

    return render_template("index.html",
                           filename=filename,
                           percent_abnormal=percent_abnormal,
                           percent_control=percent_control,
                           eeg_image=eeg_image,
                           features=features)

@app.route("/predict_depression", methods=["POST"])
def predict_depression():
    """
    Renders depression.html (Depression Results)
    """
    filename = request.form.get("filename")
    if not filename: return redirect(url_for('eeg_interface'))
    path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(path): return redirect(url_for('eeg_interface'))

    # Load Data
    raw, data, labels, fs = load_eeg(path)
    eeg_image = plot_eeg(raw)

    # Extract Features (reuse single epoch logic for demo or average logic)
    feats = extract_single_epoch_features_for_ui(data, labels, fs)
    features = {
        "alpha_O1": float(feats[0]),
        "alpha_O2": float(feats[1]),
        "coherence_O1_Fz": float(feats[2]),
        "coherence_O2_Fz": float(feats[3]),
    }

    # Calculate Depression Risk
    depression_prob = calculate_depression_score(features)
    
    return render_template("depression.html",
                           filename=filename,
                           depression_prob=depression_prob,
                           eeg_image=eeg_image,
                           features=features)

if __name__ == "__main__":
    app.run(debug=True)