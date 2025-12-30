from __future__ import annotations

import os
import sys
import re
import types
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import fitz

from flask import Flask, request, render_template, send_from_directory, Response, jsonify
from sklearn.base import BaseEstimator, TransformerMixin
from transformers import pipeline

# ============================================================================
# YOUR TENSORFLOW SPAM MODEL (EXACTLY AS IS, BUT WITH UPDATED PATHS)
# ============================================================================
# Load your TensorFlow model and tokenizer - UPDATED PATHS
model = tf.keras.models.load_model('deeplearning_models/best_spam_model.h5')
with open('deeplearning_models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
label_mapping = {'ham': 0, 'spam': 1}

# Text cleaning (must match training)
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'escapenumber|escapelong', ' ', text)
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)
    text = re.sub(r'mailto\S+|http\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s\.\?!,]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text, tokenizer, max_len=478):
    cleaned = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        sequence, maxlen=max_len, padding='post', truncating='post'
    )
    return padded

# ============================================================================
# REFERENCE FILE HELPER FUNCTIONS AND TRANSFORMERS
# ============================================================================
def _to_1d(arr):
    import numpy as _np
    return _np.asarray(arr).ravel()

def _clean_mileage(arr):
    s = pd.Series(_to_1d(arr)).astype(str).str.replace(" km", "", regex=False)
    s = pd.to_numeric(s, errors="coerce").fillna(0).clip(upper=300_000)
    return s.to_numpy(dtype=float).reshape(-1, 1)

def _clean_engine_volume(arr):
    s = pd.to_numeric(pd.Series(_to_1d(arr)), errors="coerce").fillna(0).clip(upper=6.0)
    return s.to_numpy(dtype=float).reshape(-1, 1)

def _clean_prod_year(arr):
    s = pd.to_numeric(pd.Series(_to_1d(arr)), errors="coerce").fillna(1970).clip(lower=1970)
    return s.to_numpy(dtype=float).reshape(-1, 1)

def _map_leather(arr):
    mapping = {"Yes": 1, "No": 0, "YES": 1, "NO": 0, True: 1, False: 0, "True": 1, "False": 0}
    s = pd.Series(_to_1d(arr)).map(mapping).fillna(0)
    return s.to_numpy(dtype=float).reshape(-1, 1)

def text_cleaner(X):
    s = pd.Series(X)
    s = s.astype(str).str.lower().str.strip()
    return s.apply(lambda x: re.sub(r"[^a-z\s]", "", x))

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self): self.freq_ = None
    def fit(self, X, y=None):
        s = pd.Series(_to_1d(X)); self.freq_ = s.value_counts(); return self
    def transform(self, X):
        s = pd.Series(_to_1d(X)); return s.map(self.freq_).fillna(0).to_numpy(dtype=float).reshape(-1, 1)

class BinaryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, mapping=None): self.mapping = mapping or {"No": 0, "Yes": 1}
    def fit(self, X, y=None): return self
    def transform(self, X):
        df = pd.DataFrame(X)
        for c in df.columns: df[c] = df[c].map(self.mapping)
        return df.to_numpy(dtype=float)

class OrdinalMapEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, mappings: dict | None = None): self.mappings = mappings or {}
    def fit(self, X, y=None): return self
    def transform(self, X):
        df = pd.DataFrame(X)
        for c in df.columns: df[c] = df[c].map(self.mappings.get(c, {}))
        return df.to_numpy(dtype=float)

# Register custom transformers
_fake = types.ModuleType("train_student_performance_pipeline")
for _n, _obj in {
    "BinaryEncoder": BinaryEncoder,
    "OrdinalMapEncoder": OrdinalMapEncoder,
    "FrequencyEncoder": FrequencyEncoder,
    "_to_1d": _to_1d,
    "_clean_mileage": _clean_mileage,
    "_clean_engine_volume": _clean_engine_volume,
    "_clean_prod_year": _clean_prod_year,
    "_map_leather": _map_leather,
}.items():
    setattr(_fake, _n, _obj)
sys.modules.setdefault("train_student_performance_pipeline", _fake)

_main = sys.modules["__main__"]
for _n in ["_to_1d","_clean_mileage","_clean_engine_volume","_clean_prod_year","_map_leather",
           "BinaryEncoder","OrdinalMapEncoder","FrequencyEncoder","text_cleaner"]:
    setattr(_main, _n, globals()[_n])

# ============================================================================
# FLASK APP SETUP
# ============================================================================
app = Flask(__name__, static_folder="static", template_folder="templates")

# ============================================================================
# MODEL LOADING (EXCLUDING SKLEARN SPAM MODEL) - UPDATED PATHS
# ============================================================================
def load_any(candidates: list[str], required: bool = True):
    for p in candidates:
        if os.path.exists(p):
            return joblib.load(p)
    if required:
        raise FileNotFoundError(f"None of the model files found: {candidates}")
    return None

# Load all models except sklearn spam model - UPDATED PATHS TO supervised_models/
house_model = load_any([
    "supervised_models/house_price__pipeline.joblib",
    "supervised_models/house_price_pipeline.joblib",
    "house_price_prediction_xg_model.pkl",  # Keep old path as fallback
], required=False)

car_model = load_any(["supervised_models/car_price_pipeline.joblib"], required=False)

try:
    wine_model = load_any(["supervised_models/wine_points_pipeline.joblib"], required=False)
except Exception as e:
    print(f"⚠️ Could not load wine model: {e}")
    wine_model = None

student_model = load_any(["supervised_models/student_performance_pipeline.joblib"], required=False)
cancer_model = load_any(["supervised_models/breast_cancer_pipeline.joblib"], required=False)

# PDF Summarizer (pre-trained Hugging Face model)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

# ============================================================================
# PDF SUMMARIZER FUNCTIONS
# ============================================================================
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def chunk_text(text, max_words=750):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])

# ============================================================================
# ROUTES FOR HTML PAGES
# ============================================================================
@app.get("/")
def index():
    """Main index page"""
    return render_template("index.html")

@app.get("/house")
def house_page():
    return render_template("house.html")

@app.get("/car")
def car_page():
    return render_template("car.html")

@app.get("/wine")
def wine_page():
    return render_template("wine.html")

@app.get("/student")
def student_page():
    return render_template("student.html")

@app.get("/cancer")
def cancer_page():
    return render_template("cancer.html")

@app.get("/pdf-summarizer")
def pdf_page():
    return render_template("upload_pdf.html")

# ============================================================================
# YOUR TENSORFLOW SPAM MODEL ROUTE (EXACTLY AS IS)
# ============================================================================
@app.route('/email', methods=['GET', 'POST'])
def email_predict():
    if request.method == 'GET':
        return render_template('email.html')
    
    # POST: handle prediction
    data = request.json
    email_text = data.get('text', '')
    
    processed_text = preprocess_text(email_text, tokenizer, max_len=478)
    prediction_prob = model.predict(processed_text, verbose=0)[0][0]
    
    prediction_class = 1 if prediction_prob > 0.5 else 0
    inverse_mapping = {v: k for k, v in label_mapping.items()}
    label = inverse_mapping[prediction_class]
    
    confidence = float(prediction_prob if label == 'spam' else 1 - prediction_prob)
    
    return jsonify({
        'prediction': label,
        'confidence': confidence
    })

# ============================================================================
# PREDICTION ENDPOINTS FOR ALL OTHER MODELS
# ============================================================================
def result_div(div_id: str, inner_html: str, status: int = 200) -> Response:
    return Response(f'<div class="result" id="{div_id}">{inner_html}</div>',
                    status=status, mimetype="text/html; charset=utf-8")

@app.post("/predict_house")
def predict_house():
    try:
        if house_model is None:
            return result_div("housePredictionResult", "Error: house pipeline not loaded.", status=500)
        g = lambda n: float(request.form.get(n, "0") or 0)
        X = np.array([[g("Rooms"), g("Distance"), g("Bathroom"), g("Landsize"),
                       g("BuildingArea"), g("Lattitude"), g("Longtitude"), g("Car")]])
        y = float(house_model.predict(X)[0])
        return result_div("housePredictionResult", f"🏠 Predicted House Price: ${y:,.2f}")
    except Exception as e:
        return result_div("housePredictionResult", f"Error: {e}", status=400)

@app.post("/predict_car")
def predict_car():
    try:
        if car_model is None:
            return result_div("carPredictionResult", "Error: car pipeline not loaded.", status=500)
        row = { "Manufacturer": request.form.get("Manufacturer"),
                "Model": request.form.get("Model"),
                "Prod. year": request.form.get("Prod. year"),
                "Category": request.form.get("Category"),
                "Mileage": request.form.get("Mileage"),
                "Engine volume": request.form.get("Engine volume"),
                "Leather interior": request.form.get("Leather interior"),
                "Fuel type": request.form.get("Fuel type"),
                "Gear box type": request.form.get("Gear box type"),
                "Drive wheels": request.form.get("Drive wheels"),
                "Airbags": request.form.get("Airbags") }
        X = pd.DataFrame([row], columns=["Manufacturer","Model","Prod. year","Category","Mileage",
                                         "Engine volume","Leather interior","Fuel type",
                                         "Gear box type","Drive wheels","Airbags"])
        y = float(car_model.predict(X)[0])
        return result_div("carPredictionResult", f"🚗 Predicted Car Price: ${y:,.2f}")
    except Exception as e:
        return result_div("carPredictionResult", f"Error: {e}", status=400)

@app.post("/predict_wine")
def predict_wine():
    try:
        if wine_model is None:
            return result_div("winePredictionResult", "Error: wine pipeline not loaded.", status=500)
        row = { "country": request.form.get("country"),
                "province": request.form.get("province"),
                "region_1": request.form.get("region_1"),
                "variety": request.form.get("variety"),
                "winery": request.form.get("winery"),
                "price": request.form.get("price") }
        X = pd.DataFrame([row], columns=["country","province","region_1","variety","winery","price"])
        y = float(wine_model.predict(X)[0])
        return result_div("winePredictionResult", f"🍷 Predicted Points: {y:.1f}")
    except Exception as e:
        return result_div("winePredictionResult", f"Error: {e}", status=400)

@app.post("/predict_student")
def predict_student():
    try:
        if student_model is None:
            return result_div("studentPredictionResult", "Error: student pipeline not loaded.", status=500)
        cols = [
            "Hours_Studied","Attendance","Parental_Involvement","Access_to_Resources",
            "Extracurricular_Activities","Sleep_Hours","Previous_Scores","Motivation_Level",
            "Internet_Access","Tutoring_Sessions","Family_Income","Teacher_Quality",
            "School_Type","Peer_Influence","Physical_Activity","Learning_Disabilities",
            "Parental_Education_Level","Distance_from_Home","Gender",
        ]
        numeric = {"Hours_Studied","Attendance","Sleep_Hours","Previous_Scores","Tutoring_Sessions","Physical_Activity"}
        noneable = {"Teacher_Quality","Parental_Education_Level","Distance_from_Home"}
        row: dict[str, object] = {}
        for c in cols:
            v = request.form.get(c, "")
            if c in noneable and (v is None or str(v).strip() == ""): v = None
            if c in numeric:
                try: v = float(v)
                except Exception:
                    return result_div("studentPredictionResult", f"Error: invalid numeric for {c}: {v!r}", status=400)
            row[c] = v
        X = pd.DataFrame([row], columns=cols)
        y = float(student_model.predict(X)[0])
        return result_div("studentPredictionResult", f"🎓 Predicted Performance Score: {y:.1f}")
    except Exception as e:
        return result_div("studentPredictionResult", f"Error: {e}", status=400)

@app.post("/predict_cancer")
def predict_cancer():
    try:
        if cancer_model is None:
            return result_div("cancerPredictionResult", "Error: cancer pipeline not loaded.", status=500)
        cat_in = ["Race","Marital_Status","T_Stage","N_Stage","6th_Stage","differentiate","Grade","A_Stage"]
        num_in = ["Age","Tumor_Size","Regional_Node_Examined","Reginol_Node_Positive","Survival_Months"]
        name_map = {
            "Race": "Race",
            "Marital_Status": "Marital Status",
            "T_Stage": "T Stage ",
            "N_Stage": "N Stage",
            "6th_Stage": "6th Stage",
            "differentiate": "differentiate",
            "Grade": "Grade",
            "A_Stage": "A Stage",
            "Age": "Age",
            "Tumor_Size": "Tumor Size",
            "Regional_Node_Examined": "Regional Node Examined",
            "Reginol_Node_Positive": "Reginol Node Positive",
            "Survival_Months": "Survival Months",
        }
        row_in: dict[str, object] = {}
        for c in cat_in + num_in:
            v = request.form.get(c)
            if v is None:
                return result_div("cancerPredictionResult", f"Error: missing value for {c}", status=400)
            if c in num_in:
                try: v = float(v)
                except Exception:
                    return result_div("cancerPredictionResult", f"Error: invalid numeric for {c}: {v}", status=400)
            else:
                v = str(v).strip()
            row_in[c] = v
        row_train = { name_map[k]: v for k, v in row_in.items() }
        if "Tumor Size" in row_train and row_train["Tumor Size"] is not None:
            row_train["Tumor Size"] = min(float(row_train["Tumor Size"]), 80.0)
        train_cols = [
            "Race","Marital Status","T Stage ","N Stage","6th Stage","differentiate","Grade","A Stage",
            "Age","Tumor Size","Regional Node Examined","Reginol Node Positive","Survival Months",
        ]
        X = pd.DataFrame([row_train], columns=train_cols)
        pred = cancer_model.predict(X)[0]
        return result_div("cancerPredictionResult", f"🧬 Predicted Status: {pred}")
    except Exception as e:
        return result_div("cancerPredictionResult", f"Error: {e}", status=400)

@app.route("/pdf-summarizer", methods=["GET", "POST"])
def upload_pdf():
    if request.method == "POST":
        pdf = request.files.get("pdf_file")
        if pdf:
            pdf_path = "uploaded.pdf"
            pdf.save(pdf_path)
            text = extract_text_from_pdf(pdf_path)
            if text.strip():
                summaries = []
                for chunk in chunk_text(text):
                    chunk_summary = summarizer(
                        chunk, max_length=148, min_length=128, do_sample=False
                    )[0]["summary_text"]
                    summaries.append(chunk_summary)
                combined_text = " ".join(summaries)
                final_summary = summarizer(
                    combined_text, max_length=148, min_length=128, do_sample=False
                )[0]["summary_text"]
                return jsonify({"summary": final_summary})
            else:
                return jsonify({"error": "No readable text found in the PDF."}), 400
        else:
            return jsonify({"error": "No PDF file provided."}), 400
    
    # GET request - just render the template
    return render_template("upload_pdf.html")

# ============================================================================
# STATIC FILES SERVING
# ============================================================================
@app.route("/js/<path:fname>")
def serve_js(fname):
    return send_from_directory("js", fname)

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("✅ Server starting at http://localhost:5800")
    print("📧 Spam Detection: http://localhost:5800/email")
    print("🏠 House Prediction: http://localhost:5800/house")
    print("🚗 Car Prediction: http://localhost:5800/car")
    print("🍷 Wine Prediction: http://localhost:5800/wine")
    print("🎓 Student Prediction: http://localhost:5800/student")
    print("🧬 Cancer Prediction: http://localhost:5800/cancer")
    print("📄 PDF Summarizer: http://localhost:5800/pdf-summarizer")
    app.run(debug=True, host='0.0.0.0', port=6800)