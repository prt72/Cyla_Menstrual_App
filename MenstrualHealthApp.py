import streamlit as st
import pymysql
import bcrypt
import pandas as pd
import xgboost as xgb
import tempfile
import logging
import numpy as np
import joblib
import re
import xgboost as xgb
import os
import tempfile
from fpdf import FPDF
from datetime import datetime
from streamlit_scroll_to_top import scroll_to_here
from sklearn.exceptions import NotFittedError
from email_utils import send_email_with_attachment
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access them using os.getenv
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
PASSWORD_APP = os.getenv("PASSWORD_APP")

# ------------------ BASIC LOGGING (IMPROVEMENT) ------------------
LOGFILE = os.getenv("APP_LOGFILE", "app.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGFILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ------------------ APP CONFIG ------------------
st.set_page_config(
    page_title="Cyla",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ GLOBAL CSS ------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

/* Sidebar Styles */
[data-testid="stSidebar"] {
    position: relative;
    background-image: url("https://i.imgur.com/9PA7Upa.jpeg");
    background-size: cover;
    background-repeat: no-repeat;
    background-position: center;
    border-right: 3px solid #ff99aa;
    padding: 15px;
    color: black;
    width: 350px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

/* Overlay to lighten background image */
[data-testid="stSidebar"]::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(255, 255, 255, 0.5);
    z-index: 0;
    border-right: 3px solid #ff99aa;
}

/* Ensure sidebar content is above overlay */
[data-testid="stSidebar"] * {
    position: relative;
    z-index: 1;
}
/* Sidebar Logout Button */
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(90deg, #ED778A, #ffd6e0);
    color: #7e0000;
    font-weight: 700;
    border: none;
    border-radius: 30px;
    padding: 0.8rem 2rem;
    margin-top: 1.2rem;
    font-size: 1.1rem;
    transition: all 0.3s;
    box-shadow: 0 4px 12px rgba(255, 179, 198, 0.18);
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(90deg, #ffb3c6, #ffd6e0);
    box-shadow: 0 6px 18px rgba(255, 179, 198, 0.28);
    transform: translateY(-2px);
}            
/* Phase & Risk Styles */
.main > div {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(15px);
    border-radius: 25px;
    padding: 2rem;
    margin: 1rem 0;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

/* === ENHANCED TITLE === */
h1 {
    font-size: 2.9rem !important;
    background: linear-gradient(135deg, #7e0000, #c41e3a, #e91e63, #ff69b4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    font-weight: 800;
    letter-spacing: -1px;
    text-shadow: 2px 2px 8px rgba(0,0,0,0.1);
    animation: fadeInDown 1.2s ease-out;
    position: relative;
}
            
h1, h2, h3 {
    color: #7e0000 !important;
    font-weight: 600;
}

.section-card {
    background: linear-gradient(145deg, #ffccd5, #ff99aa);
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.3);
    animation: fadeIn 1s ease-in;
}

.menstrual-phase {
    background: linear-gradient(145deg, #ff9a9e, #fad0c4);
    color: black;
    border-radius: 20px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 10px 30px rgba(255, 107, 107, 0.3);
    text-align: center;
}

.follicular-phase {
    background: linear-gradient(145deg, #a1c4fd, #c2e9fb);
    color: black;
    border-radius: 20px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 10px 30px rgba(161, 196, 253, 0.3);
    text-align: center;
}

.ovulation-phase {
    background: linear-gradient(145deg, #ffecd2, #fcb69f);
    color: black;
    border-radius: 20px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 10px 30px rgba(252, 182, 159, 0.3);
    text-align: center;
}

.luteal-phase {
    background: linear-gradient(145deg, #d4fc79, #96e6a1);
    color: black;
    border-radius: 20px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 10px 30px rgba(148, 252, 121, 0.3);
    text-align: center;
}

.menstrual-phase h2,
.follicular-phase h2,
.ovulation-phase h2,
.luteal-phase h2 {
    background: none !important;
    -webkit-background-clip: unset !important;
    -webkit-text-fill-color: #000000 !important;
    background-clip: unset !important;
    color: #000000 !important;
}
            
.recommendation-card {
    background: white;
    border-radius: 20px;
    padding: 1.5rem;
    margin: 1rem 0;
    border: 1px solid #ff99aa;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    animation: fadeIn 1s ease-in;
}

.recommendation-card h4 {
    color: #7e0000;
    margin-bottom: 1rem;
    font-weight: 600;
}

.recommendation-card p {
    color: black;
    line-height: 1.6;
    margin-bottom: 0.5rem;
}

/* Hide default browser password reveal/icons */
input[type="password"]::-ms-reveal,
input[type="password"]::-ms-clear {
    display: none;
}

/* Global styling for consistency across pages */
.main-container {
    background: linear-gradient(135deg, #fef6f8, #f9e4e8);
    padding: 2.5rem;
    border-radius: 20px;
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
    margin: 1.5rem 0;
}
/* Headers */
h1, h2, h3, h4 {
    color: #8b0026;
    font-weight: 600;
    margin-bottom: 1.2rem;
}
h2 {
    font-size: 2.2rem;
    text-align: center;
    background: linear-gradient(45deg, #ff99aa, #ffccd5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
h3 {
    font-size: 1.7rem;
    border-bottom: 3px solid #ff99aa;
    padding-bottom: 0.6rem;
}
/* Input fields */
.stNumberInput, .stRadio, .stSelectbox {
    background-color: #fff;
    border-radius: 12px;
    padding: 0.6rem;
    border: 2px solid #ffd1d9;
    transition: all 0.3s ease;
}
.stNumberInput:hover, .stRadio:hover, .stSelectbox:hover {
    border-color: #ff99aa;
    box-shadow: 0 0 10px rgba(255, 153, 170, 0.4);
}
.stNumberInput label, .stRadio label, .stSelectbox label {
    color: #8b0026;
    font-weight: 500;
}

/* Info and symptom boxes */
.symptom-box, .info-box {
    background: linear-gradient(145deg, #fff0f3, #ffe4e8);
    padding: 1.8rem;
    border-radius: 15px;
    margin: 1.2rem 0;
    color: #333;
    font-size: 1.05rem;
    box-shadow: 0 3px 8px rgba(0, 0, 0, 0.05);
}
.symptom-box p {
    font-weight: 500;
    margin-bottom: 0.8rem;
}
.symptom-box ul {
    list-style-type: none;
    padding-left: 0;
}
.symptom-box li {
    padding: 0.6rem 0;
    font-size: 1.05rem;
}
.symptom-box li::before {
    content: "🌸 ";
    color: #ff6680;
}
/* Recommendations styling */
.recommendation-box {
    padding: 2.5rem;
    border-radius: 20px;
    margin: 2.5rem 0;
    animation: slideIn 0.6s ease-out;
    color: #333;
}
.high-risk {
    background: linear-gradient(145deg, #ffebee, #ffcdd2);
    border-left: 6px solid #d32f2f;
}
.moderate-risk {
    background: linear-gradient(145deg, #fff8e1, #ffecb3);
    border-left: 6px solid #ef6c00;
}
.low-risk {
    background: linear-gradient(145deg, #e0f2f1, #b2dfdb);
    border-left: 6px solid #00796b;
}
.recommendation-box h4 {
    margin-top: 0;
    font-size: 1.6rem;
    color: inherit;
}
.recommendation-box p {
    margin: 0.8rem 0;
    line-height: 1.7;
    font-size: 1.05rem;
}
.recommendation-box ul {
    padding-left: 1.8rem;
    margin: 1.2rem 0;
}
.recommendation-box li {
    margin-bottom: 0.6rem;
    font-size: 1.05rem;
}
/* Animation for recommendations */
@keyframes slideIn {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}
/* Warning box */
.warning-box {
    text-align: center;
    padding: 2.5rem;
    border-radius: 20px;
    border: 3px solid #ffa000;
    margin: 2.5rem 0;
    animation: fadeIn 0.5s ease-in;
}
.warning-box h3 {
    color: #ef6c00;
    font-size: 1.8rem;
    margin: 0;
}
/* Tab styling */
.stTabs [data-baseweb="tab"] {
    background-color: #fff;
    border-radius: 12px 12px 0 0;
    padding: 0.9rem 2rem;
    margin-right: 0.6rem;
    color: #8b0026;
    font-weight: 600;
    font-size: 1.1rem;
}
.stTabs [data-baseweb="tab"]:hover {
    background-color: #ffd1d9;
    color: #8b0026;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #ff99aa, #ff6680);
    color: white;
}
/* Ensure Streamlit info box is styled */
.stAlert {
    border-radius: 12px;
    padding: 1.5rem;
    color: #0d47a1;
}
/* Reduce font size of sidebar "Navigation" title */
[data-testid="stSidebar"] h1 {
    font-size: 2.5rem !important;
    margin-bottom: 0.5rem !important;
}
[data-testid="stSidebar"] .stSuccess {
    font-size: 0.85rem !important;
    padding: 0.7rem 1rem !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------ DATABASE FUNCTIONS ------------------
def get_db_connection():
    try:
        # Write SSL cert to temp file from secrets (similar to what we discussed earlier)
        cert_str = st.secrets["MYSQL_SSL_CA_PEM"]
        with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".crt") as f:
            f.write(cert_str.replace("\\n", "\n"))
            cert_file_path = f.name

        connection = pymysql.connect(
            host=st.secrets["MYSQL_HOST"],
            user=st.secrets["MYSQL_USER"],
            password=st.secrets["MYSQL_PASS"],
            db=st.secrets["MYSQL_DB"],
            port=int(st.secrets["MYSQL_PORT"]),
            cursorclass=pymysql.cursors.DictCursor,
            ssl={'ca': cert_file_path}
        )
        return connection
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None



def create_user(username: str, email: str, password: str) -> int: # Return type hint changed
    conn = get_db_connection()
    if not conn:
        return None
    try:
        with conn.cursor() as cursor:
            # Check if user exists
            cursor.execute("SELECT user_id FROM users WHERE username=%s OR email=%s", (username, email))
            if cursor.fetchone():
                return None 

            pw_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            cursor.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
                (username, email, pw_hash.decode('utf-8'))
            )
            conn.commit()
            # --- Get the ID of the newly inserted user ---
            new_user_id = cursor.lastrowid # This gets the auto-generated ID from the last INSERT
            return new_user_id # Return the REAL user_id (integer)
    except Exception as e:
        st.error(f"Error creating user: {e}")
        return None 
    finally:
        conn.close()

def verify_user(username: str, password: str):
    conn = get_db_connection()
    if not conn:
        return False, None
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT user_id, password_hash FROM users WHERE username=%s", (username,))
            row = cursor.fetchone()
            if row and bcrypt.checkpw(password.encode('utf-8'), row['password_hash'].encode('utf-8')):
                return True, row['user_id']
            return False, None
    except Exception as e:
        st.error(f"Error verifying user: {e}")
        return False, None
    finally:
        conn.close()

def get_user_email(user_id):
    conn = get_db_connection()
    if not conn:
        return None
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT email FROM users WHERE user_id=%s", (user_id,))
            row = cursor.fetchone()
            return row['email'] if row else None
    except Exception as e:
        st.error(f"Error retrieving email: {e}")
        return None
    finally:
        conn.close()

def save_symptom_input_to_db(user_id, hair_growth, weight_gain, skin_darkening, fast_food, cycle_length):
    conn = get_db_connection()
    if not conn:
        return
    try:
        with conn.cursor() as cursor:
            query = """
            INSERT INTO symptom_inputs 
                (user_id, hair_growth, weight_gain, skin_darkening, fast_food, cycle_length, submitted_at)
            VALUES 
                (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            """
            cursor.execute(query, (
                int(user_id),
                str(hair_growth),
                str(weight_gain),
                str(skin_darkening),
                str(fast_food),
                int(cycle_length)
            ))
            conn.commit()
    except Exception as e:
        st.error(f"Database error: {e}")
    finally:
        conn.close()

def get_symptom_history(user_id):
    conn = get_db_connection()
    if not conn:
        return []
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM symptom_inputs WHERE user_id=%s ORDER BY submitted_at DESC", (user_id,))
            return cursor.fetchall()
    except Exception as e:
        st.error(f"Error retrieving history: {e}")
        return []
    finally:
        conn.close()

# Function to save Page 3 inputs to detailed_inputs table
def save_detailed_input_to_db(user_id, nc_inputs, clinical_inputs, category):
    conn = get_db_connection()
    if not conn:
        return
    try:
        with conn.cursor() as cursor:
            columns = [
                'user_id', 'age', 'weight', 'height', 'bmi', 'cycle_length', 'marriage_years',
                'hip', 'waist', 'wh_ratio', 'pregnant', 'abortions', 'weight_gain', 'hair_growth',
                'skin_darkening', 'hair_loss', 'pimples', 'fast_food', 'exercise',
                'blood_group', 'pulse_rate', 'rr_breaths', 'hb', 'fsh', 'lh', 'tsh', 'amh',
                'prl', 'vit_d3', 'prg', 'rbs', 'systolic_bp', 'diastolic_bp', 'follicle_l',
                'follicle_r', 'fsize_l', 'fsize_r', 'endometrium', 'risk_category'
            ]
            values = {
                'user_id': user_id,
                'age': nc_inputs['Age (yrs)'],
                'weight': nc_inputs['Weight (Kg)'],
                'height': nc_inputs['Height(Cm)'],
                'bmi': nc_inputs['BMI'],
                'cycle_length': nc_inputs['Cycle length(days)'],
                'marriage_years': nc_inputs['Marriage Status (Yrs)'],
                'hip': nc_inputs['Hip(inch)'],
                'waist': nc_inputs['Waist(inch)'],
                'wh_ratio': nc_inputs['Waist:Hip Ratio'],
                'pregnant': nc_inputs['Pregnant(Y/N)'],
                'abortions': nc_inputs['No. of Abortions'],
                'weight_gain': nc_inputs['Weight gain(Y/N)'],
                'hair_growth': nc_inputs['hair growth(Y/N)'],
                'skin_darkening': nc_inputs['Skin darkening (Y/N)'],
                'hair_loss': nc_inputs['Hair loss(Y/N)'],
                'pimples': nc_inputs['Pimples(Y/N)'],
                'fast_food': nc_inputs['Fast food (Y/N)'],
                'exercise': nc_inputs['Reg.Exercise(Y/N)'],
                'blood_group': clinical_inputs.get('Blood Group', None),
                'pulse_rate': clinical_inputs.get('Pulse rate(bpm)', None),
                'rr_breaths': clinical_inputs.get('RR (breaths/min)', None),
                'hb': clinical_inputs.get('Hb(g/dl)', None),
                'fsh': clinical_inputs.get('FSH(mIU/mL)', None),
                'lh': clinical_inputs.get('LH(mIU/mL)', None),
                'tsh': clinical_inputs.get('TSH (mIU/L)', None),
                'amh': clinical_inputs.get('AMH(ng/mL)', None),
                'prl': clinical_inputs.get('PRL(ng/mL)', None),
                'vit_d3': clinical_inputs.get('Vit D3 (ng/mL)', None),
                'prg': clinical_inputs.get('PRG(ng/mL)', None),
                'rbs': clinical_inputs.get('RBS(mg/dl)', None),
                'systolic_bp': clinical_inputs.get('BP _Systolic (mmHg)', None),
                'diastolic_bp': clinical_inputs.get('BP _Diastolic (mmHg)', None),
                'follicle_l': clinical_inputs.get('Follicle No. (L)', None),
                'follicle_r': clinical_inputs.get('Follicle No. (R)', None),
                'fsize_l': clinical_inputs.get('Avg. F size (L) (mm)', None),
                'fsize_r': clinical_inputs.get('Avg. F size (R) (mm)', None),
                'endometrium': clinical_inputs.get('Endometrium (mm)', None),
                'risk_category': category,
            }
            placeholders = ', '.join(['%s'] * len(columns))
            columns_str = ', '.join(columns)
            sql = f"INSERT INTO detailed_inputs ({columns_str}) VALUES ({placeholders})"
            cursor.execute(sql, tuple(values.values()))
            conn.commit()
    except Exception as e:
        st.error(f"Database error: {e}")
    finally:
        conn.close()

def is_valid_email(email):
    """Check if email has valid format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def check_user_exists(username=None, email=None):
    """Check if username or email already exists in database"""
    conn = get_db_connection()
    if not conn:
        return False  # Assume doesn't exist if we can't check
    
    try:
        with conn.cursor() as cursor:
            if username and email:
                # Check both username AND email (for registration)
                cursor.execute(
                    "SELECT user_id FROM users WHERE username = %s OR email = %s", 
                    (username, email)
                )
            elif username:
                # Check only username (for login)
                cursor.execute(
                    "SELECT user_id FROM users WHERE username = %s", 
                    (username,)
                )
            elif email:
                # Check only email
                cursor.execute(
                    "SELECT user_id FROM users WHERE email = %s", 
                    (email,)
                )
            else:
                return False
                
            return cursor.fetchone() is not None
    except Exception as e:
        st.error(f"Error checking user existence: {e}")
        return False  # Assume doesn't exist if error occurs
    finally:
        conn.close()

# --- FIXED: Always point to the correct models folder ---
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Features lists (unchanged) ---
non_clinical_features = [
        'Age (yrs)', 'Weight (Kg)', 'Height(Cm)', 'BMI', 'Cycle length(days)',
        'Marriage Status (Yrs)', 'Hip(inch)', 'Waist(inch)', 'Waist:Hip Ratio',
        'Pregnant(Y/N)', 'No. of Abortions', 'Weight gain(Y/N)', 'hair growth(Y/N)',
        'Skin darkening (Y/N)', 'Hair loss(Y/N)', 'Pimples(Y/N)', 'Fast food (Y/N)',
        'Reg.Exercise(Y/N)'
    ]
overall_features = [
        'Age (yrs)', 'Weight (Kg)', 'Height(Cm)', 'BMI', 'Blood Group', 'Pulse rate(bpm)', 
        'RR (breaths/min)', 'Hb(g/dl)', 'Cycle length(days)', 'Marriage Status (Yrs)', 
        'Pregnant(Y/N)', 'No. of Abortions', 'FSH(mIU/mL)', 'LH(mIU/mL)', 'FSH/LH', 
        'Hip(inch)', 'Waist(inch)', 'Waist:Hip Ratio', 'TSH (mIU/L)', 'AMH(ng/mL)', 
        'PRL(ng/mL)', 'Vit D3 (ng/mL)', 'PRG(ng/mL)', 'RBS(mg/dl)', 'Weight gain(Y/N)', 
        'hair growth(Y/N)', 'Skin darkening (Y/N)', 'Hair loss(Y/N)', 'Pimples(Y/N)', 
        'Fast food (Y/N)', 'Reg.Exercise(Y/N)', 'BP _Systolic (mmHg)', 'BP _Diastolic (mmHg)', 
        'Follicle No. (L)', 'Follicle No. (R)', 'Avg. F size (L) (mm)', 'Avg. F size (R) (mm)', 
        'Endometrium (mm)'
    ]

class XGBoosterWrapper:
    def __init__(self, booster: xgb.core.Booster):
        self.booster = booster

    def predict_proba(self, X):
        if hasattr(X, "values"):
            arr = X.values
        else:
            arr = np.asarray(X)
        dmat = xgb.DMatrix(arr)
        preds = self.booster.predict(dmat)
        preds = np.array(preds)
        if preds.ndim == 1:
            return np.vstack([1 - preds, preds]).T
        return preds

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

class ModelFactory:
    _models = {}

    @staticmethod
    def get_model(model_type: str):
        model_type = str(model_type).lower()
        if model_type in ModelFactory._models:
            return ModelFactory._models[model_type]

        # Define model paths
        model_paths = {
            "all": os.path.join(MODEL_DIR, "ALL_XGB_model.joblib"),
            "nc": os.path.join(MODEL_DIR, "NC_XGB_model.joblib")
        }

        if model_type not in model_paths:
            st.error(f"Invalid model type: {model_type}")
            return None

        path = model_paths[model_type]
        if not os.path.exists(path):
            st.error(f"Model file not found: {path}")
            return None

        try:
            # Try loading with joblib first
            loaded = joblib.load(path)
        except Exception as e_joblib:
            # If joblib fails, try loading as native XGBoost Booster
            try:
                booster = xgb.Booster()
                booster.load_model(path)
                wrapper = XGBoosterWrapper(booster)
                ModelFactory._models[model_type] = wrapper
                return wrapper
            except Exception as e_booster:
                st.error(f"Failed to load model: joblib error: {e_joblib}; xgboost error: {e_booster}")
                return None

        # Handle different types of loaded objects
        if hasattr(loaded, "predict_proba") and callable(loaded.predict_proba):
            # It's a scikit-learn compatible model
            if hasattr(loaded, "classes_"):
                # ✅ Model is fitted
                ModelFactory._models[model_type] = loaded
                return loaded
            else:
                st.error("❌ Loaded model has no classes_. It is not fitted.")
                return None

        elif isinstance(loaded, xgb.core.Booster):
            # Wrap native XGBoost Booster
            wrapper = XGBoosterWrapper(loaded)
            ModelFactory._models[model_type] = wrapper
            return wrapper

        else:
            st.error(f"Unsupported model type: {type(loaded)}")
            return None
            
# ------------------ UTILITY FUNCTIONS ------------------
def get_menstrual_phase(lmp_date: datetime, today: datetime) -> str:
    days_since_lmp = (today - lmp_date).days
    cycle_day = days_since_lmp % 28
    if cycle_day <= 4:
        return "Menstrual Phase"
    elif 5 <= cycle_day <= 13:
        return "Follicular Phase"
    elif 14 <= cycle_day <= 16:
        return "Ovulation Phase"
    else:
        return "Luteal Phase"

def get_phase_recommendation(phase: str) -> dict:
    recs = {
        "Menstrual Phase": {
            "title": "Menstrual Phase - Time to Rest & Restore",
            "wellness": [
                "💤 Prioritize 8-9 hours of sleep for optimal recovery",
                "🛁 Take warm baths with Epsom salts to ease cramps",
                "🧘‍♀️ Practice gentle yoga or meditation for 15-20 minutes",
                "🌿 Try herbal teas like chamomile or ginger for comfort"
            ],
            "nutrition": [
                "🥬 Load up on iron-rich foods: spinach, lentils, dark chocolate",
                "🍌 Eat potassium-rich foods: bananas, avocados, sweet potatoes",
                "🥛 Increase calcium intake: dairy, leafy greens, almonds",
                "💧 Stay hydrated with water infused with lemon or cucumber"
            ],
            "exercise": [
                "🚶‍♀️ Gentle walks in nature (20-30 minutes)",
                "🧘‍♀️ Restorative yoga or yin yoga",
                "🏊‍♀️ Light swimming if comfortable",
                "🛌 Listen to your body - rest when needed"
            ],
            "mood": [
                "📚 Read uplifting books or magazines",
                "🎵 Listen to calming music or nature sounds",
                "💆‍♀️ Treat yourself to a gentle self-massage",
                "🌸 Practice gratitude journaling"
            ]
        },
        "Follicular Phase": {
            "title": "Follicular Phase - New Beginnings & Fresh Energy",
            "wellness": [
                "⭐ Perfect time to start new habits or projects",
                "🌅 Wake up earlier to maximize your natural energy",
                "🧴 Establish a consistent skincare routine",
                "📱 Try new wellness apps or meditation practices"
            ],
            "nutrition": [
                "🥗 Focus on fresh, colorful vegetables and fruits",
                "🐟 Include lean proteins: fish, chicken, tofu, beans",
                "🥑 Add healthy fats: avocados, nuts, olive oil",
                "🌾 Choose complex carbs: quinoa, brown rice, oats"
            ],
            "exercise": [
                "🏃‍♀️ Cardio workouts: running, cycling, dancing",
                "💪 Strength training with moderate weights",
                "🤸‍♀️ Try new fitness classes or activities",
                "🏃‍♀️ Aim for 30-45 minutes of activity daily"
            ],
            "mood": [
                "🎯 Set new goals and create action plans",
                "📚 Learn something new or take up a hobby",
                "👥 Socialize and connect with friends",
                "🌟 Practice positive affirmations daily"
            ]
        },
        "Ovulation Phase": {
            "title": "Ovulation Phase - Peak Power & Confidence",
            "wellness": [
                "🔥 Your energy is at its highest - embrace it!",
                "💄 You naturally look more radiant during this phase",
                "🗣️ Great time for important conversations or presentations",
                "💃 Your coordination and physical performance peak"
            ],
            "nutrition": [
                "🥜 Boost zinc intake: pumpkin seeds, cashews, beef",
                "🍇 Antioxidant-rich foods: berries, dark grapes, green tea",
                "🥚 B-vitamin rich foods: eggs, leafy greens, nutritional yeast",
                "🌿 Anti-inflammatory foods: turmeric, ginger, fatty fish"
            ],
            "exercise": [
                "🏋️‍♀️ High-intensity workouts and strength training",
                "🏃‍♀️ Sprint intervals and plyometric exercises",
                "🤸‍♀️ Try challenging new workouts or sports",
                "🕺 Dance workouts or energetic group classes"
            ],
            "mood": [
                "👑 Embrace your natural confidence and charisma",
                "💼 Schedule important meetings or interviews",
                "🎉 Plan social events or date nights",
                "🌟 Take on leadership roles or new challenges"
            ]
        },
        "Luteal Phase": {
            "title": "Luteal Phase - Time to Slow Down & Nurture",
            "wellness": [
                "🛌 Prioritize rest and avoid overcommitting",
                "🌙 Create calming bedtime routines",
                "🧘‍♀️ Practice stress-reduction techniques",
                "🌿 Use aromatherapy with lavender or bergamot"
            ],
            "nutrition": [
                "🍫 Satisfy cravings with dark chocolate (70%+ cacao)",
                "🥔 Complex carbs for serotonin: sweet potatoes, oats",
                "🥜 Magnesium-rich foods: almonds, spinach, pumpkin seeds",
                "🌿 Herbal teas: red raspberry leaf, evening primrose"
            ],
            "exercise": [
                "🚶‍♀️ Gentle walks or light hiking",
                "🧘‍♀️ Yoga focusing on hip openers and twists",
                "🏊‍♀️ Swimming or water aerobics",
                "🕺 Low-impact dance or pilates"
            ],
            "mood": [
                "📝 Journal your thoughts and emotions",
                "🎨 Engage in creative activities: art, music, writing",
                "🛁 Take relaxing baths with essential oils",
                "🤗 Practice self-compassion and patience"
            ]
        }
    }
    return recs.get(phase, {
        "title": "Unable to determine phase",
        "wellness": ["Please try again with a valid date"],
        "nutrition": [],
        "exercise": [],
        "mood": []
    })

def display_enhanced_recommendations(phase_data):
    phase_class = ""
    if "Menstrual" in phase_data['title']:
        phase_class = "menstrual-phase"
    elif "Follicular" in phase_data['title']:
        phase_class = "follicular-phase"
    elif "Ovulation" in phase_data['title']:
        phase_class = "ovulation-phase"
    elif "Luteal" in phase_data['title']:
        phase_class = "luteal-phase"
    
    st.markdown(f"""
    <div class="{phase_class}">
        <h2>{phase_data['title']}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        wellness_html = """
        <div class="recommendation-card">
            <h4>🌟 Wellness & Self-Care</h4>
        """
        for tip in phase_data['wellness']:
            wellness_html += f"<p style='color: black; margin: 0.5rem 0;'>{tip}</p>"
        wellness_html += "</div>"
        st.markdown(wellness_html, unsafe_allow_html=True)
        
        nutrition_html = """
        <div class="recommendation-card">
            <h4>🍎 Nutrition Focus</h4>
        """
        for tip in phase_data['nutrition']:
            nutrition_html += f"<p style='color: black; margin: 0.5rem 0;'>{tip}</p>"
        nutrition_html += "</div>"
        st.markdown(nutrition_html, unsafe_allow_html=True)
    
    with col2:
        exercise_html = """
        <div class="recommendation-card">
            <h4>💪 Exercise & Movement</h4>
        """
        for tip in phase_data['exercise']:
            exercise_html += f"<p style='color: black; margin: 0.5rem 0;'>{tip}</p>"
        exercise_html += "</div>"
        st.markdown(exercise_html, unsafe_allow_html=True)
        
        mood_html = """
        <div class="recommendation-card">
            <h4>🧠 Mood & Mindset</h4>
        """
        for tip in phase_data['mood']:
            mood_html += f"<p style='color: black; margin: 0.5rem 0;'>{tip}</p>"
        mood_html += "</div>"
        st.markdown(mood_html, unsafe_allow_html=True)

def validate_date(selected_date):
    today = datetime.today().date()
    if selected_date > today:
        st.error("You cannot select a future date. Please choose today's date or a previous date.")
        return today
    return selected_date

def generate_pdf_report(symptom_data, risk_category, username):
    class PDF(FPDF):
        def header(self):
            # Simple header with color
            self.set_fill_color(70, 130, 180)  # Steel blue
            self.rect(0, 0, 210, 20, 'F')  # Colored header background

            self.set_text_color(255, 255, 255)
            self.set_font('Arial', 'B', 16)

            self.set_y(6)  # Position the text vertically within the blue bar
            self.cell(0, 10, 'PCOS Risk Assessment Report', 0, 1, 'C')

            self.set_text_color(0, 0, 0)
            self.ln(10)


        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, f'Generated on {datetime.now().strftime("%B %d, %Y")} - This is not medical advice', 0, 0, 'C')

    # Create PDF
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Patient Information
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(70, 130, 180)
    pdf.cell(0, 10, 'Patient Information', 0, 1, 'L')
    pdf.ln(2)

    pdf.set_font('Arial', '', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(30, 8, 'Userame:', 0, 0)
    pdf.cell(0, 8, username, 0, 1)
    pdf.cell(30, 8, 'Date:', 0, 0)
    pdf.cell(0, 8, datetime.now().strftime("%B %d, %Y"), 0, 1)
    pdf.ln(10)

    # Risk Assessment Result
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(70, 130, 180)
    pdf.cell(0, 10, 'Risk Assessment Result', 0, 1, 'L')
    pdf.ln(2)

    # Risk category with colored background
    if 'high' in risk_category.lower():
        pdf.set_fill_color(220, 53, 69)  # Red
        pdf.set_text_color(255, 255, 255)
    elif 'moderate' in risk_category.lower():
        pdf.set_fill_color(255, 193, 7)  # Orange
        pdf.set_text_color(0, 0, 0)
    else:
        pdf.set_fill_color(40, 167, 69)  # Green
        pdf.set_text_color(255, 255, 255)

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, f'Risk Level: {risk_category.upper()}', 1, 1, 'C', True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)

    # Explanation
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(70, 130, 180)
    pdf.cell(0, 10, 'Explanation', 0, 1, 'L')
    pdf.ln(2)

    risk_explanations = {
        'low': "Your responses suggest a lower likelihood of PCOS. Continue monitoring your health and maintain regular check-ups with your healthcare provider.",
        'moderate': "Your responses indicate some symptoms that may be associated with PCOS. We recommend consulting with a healthcare professional for further evaluation.",
        'high': "Your responses suggest several symptoms commonly associated with PCOS. Please consult with a healthcare professional promptly for proper evaluation."
    }

    explanation = risk_explanations.get(risk_category.lower(), "Please consult with a healthcare professional for proper evaluation.")
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, explanation, 0, 'J')
    pdf.ln(10)

    # Symptom Analysis
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(70, 130, 180)
    pdf.cell(0, 10, 'Your Responses', 0, 1, 'L')
    pdf.ln(2)

    # Table header
    pdf.set_font('Arial', 'B', 11)
    pdf.set_fill_color(240, 240, 240)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(90, 8, 'Symptom/Factor', 1, 0, 'C', True)
    pdf.cell(90, 8, 'Response', 1, 1, 'C', True)

    # Table content
    pdf.set_font('Arial', '', 10)
    for i, (key, value) in enumerate(symptom_data.items()):
        if i % 2 == 0:
            pdf.set_fill_color(250, 250, 250)
        else:
            pdf.set_fill_color(255, 255, 255)
        
        # Format the key
        formatted_key = key.replace('_', ' ').title()
        
        # Format the value
        if isinstance(value, bool):
            formatted_value = "Yes" if value else "No"
        else:
            formatted_value = str(value)
        
        pdf.cell(90, 6, formatted_key, 1, 0, 'L', True)
        pdf.cell(90, 6, formatted_value, 1, 1, 'L', True)

    pdf.ln(10)

    # Recommendations
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(40, 167, 69)
    pdf.cell(0, 10, 'General Health Recommendations', 0, 1, 'L')
    pdf.ln(2)

    recommendations = [
        "Maintain a balanced diet rich in whole foods and low in processed sugars",
        "Engage in regular physical activity (aim for 150 minutes per week)",
        "Maintain a healthy weight through proper nutrition and exercise",
        "Manage stress through relaxation techniques and adequate sleep (7-9 hours)",
        "Keep track of your menstrual cycles and symptoms",
        "Schedule regular check-ups with your healthcare provider",
        "Consider consulting specialists if symptoms persist or worsen"
    ]

    pdf.set_font('Arial', '', 11)
    pdf.set_text_color(0, 0, 0)
    for i, rec in enumerate(recommendations, 1):
        pdf.cell(8, 6, f'{i}.', 0, 0)
        pdf.multi_cell(0, 6, rec, 0, 'L')
        pdf.ln(1)

    pdf.ln(5)

    # Disclaimer
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(220, 53, 69)
    pdf.cell(0, 10, 'IMPORTANT MEDICAL DISCLAIMER', 0, 1, 'C')
    pdf.ln(2)

    disclaimer = (
        "This assessment is a screening tool only and is NOT a medical diagnosis. "
        "PCOS can only be properly diagnosed by qualified healthcare professionals "
        "through comprehensive medical evaluation. If you have health concerns, "
        "please consult with your doctor, gynecologist, or endocrinologist."
    )

    pdf.set_font('Arial', '', 10)
    pdf.set_text_color(0, 0, 0)
    pdf.set_fill_color(255, 240, 240)
    pdf.multi_cell(0, 6, disclaimer, 1, 'J', True)

    # Save to temporary file and return the file path
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    pdf.output(temp_file.name)
    
    return temp_file.name

# Utility to check if an email is valid
def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

# Modified handle_report_output 
def handle_report_output(symptom_data, risk_category, username, user_id, page_key):
    st.markdown("""
    <h3 style='text-align:center; color:#7e0000; font-size:1.4rem; font-weight:700; margin-bottom:1.2rem;'>
        📋 Get Your PCOS Risk Assessment Report
    </h3>
    """, unsafe_allow_html=True)

    try:
        pdf_file_path = generate_pdf_report(symptom_data, risk_category, username)

        with open(pdf_file_path, 'rb') as pdf_file:
            pdf_data = pdf_file.read()

        os.unlink(pdf_file_path)

        # Add horizontal radio button to choose action
        report_option = st.radio(
            "Choose how you'd like to receive your report:",
            ["Email Report", "Download PDF"],
            key=f"report_option_{page_key}",
            horizontal=True
        )

        if report_option == "Email Report":
            st.subheader("📧 Email Report")
            with st.form(f"email_form_{page_key}"):
                email = st.text_input("Enter your email address:")
                submit_email = st.form_submit_button("Send Report", use_container_width=True)

                if submit_email:
                    if email and '@' in email and '.' in email:
                        try:
                            success = send_email_with_attachment(
                                sender_email=st.secrets["EMAIL"]["EMAIL_SENDER"],
                                app_password=st.secrets["EMAIL"]["PASSWORD_APP"],
                                receiver_email=email,
                                subject="Your PCOS Risk Assessment Report",
                                body_text=f"Dear {username},\n\nPlease find your personalized PCOS risk assessment report attached.\n\nRemember: This is for informational purposes only and not medical advice. Please consult with healthcare professionals for proper medical evaluation.\n\nStay healthy! 💖",
                                attachment_bytes=pdf_data,
                                filename=f"PCOS_Risk_Report.pdf"
                            )
                            if success:
                                st.success(f"✅ Report sent successfully to {email}!")
                            else:
                                st.error("❌ Failed to send email. Please try again.")
                        except Exception as e:
                            st.error(f"❌ Error sending email: {str(e)}")
                    else:
                        st.error("❌ Please enter a valid email address.")

        elif report_option == "Download PDF":
            st.subheader("⬇️ Download Report")
            st.markdown("<br>", unsafe_allow_html=True)
            filename = f"PCOS_Risk_Report.pdf"
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_data,
                file_name=filename,
                mime="application/pdf",
                use_container_width=True
            )

        st.success("✅ Your report has been generated successfully!")

    except Exception as e:
        st.error(f"❌ Error generating report: {str(e)}")
        st.info("Please try again or contact support if the problem persists.")

# ------------------ PAGE FUNCTIONS ------------------

# --- Logger for model loading ---
logger = logging.getLogger(__name__)

# Login Page
def page_login_register():
    # ENHANCED STYLING: Soft Pink Gradient, Logo, Features, Glow
    st.markdown("""
    <style>
    /* === EVEN LIGHTER SOFT PINK GRADIENT BACKGROUND === */
    .stApp {
        background: linear-gradient(135deg, #ffffff 0%, #fff8fa 25%, #ffeef5 50%, #fff0f6 75%, #ffffff 100%);
        background-attachment: fixed;
        font-family: 'Quicksand', sans-serif, 'Helvetica Neue', sans-serif;
        color: #333;
        min-height: 100vh;
    }

    .main {
        position: relative;
        z-index: 1;
        padding-top: 2rem;
    }

    /* === ENHANCED LOGO CIRCLE === */
    .logo-circle {
        width: 200px;
        height: 200px;
        border-radius: 50%;
        margin: 1.5rem auto 0rem;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 
            0 8px 32px rgba(255, 105, 180, 0.2),
            0 2px 8px rgba(0,0,0,0.1),
            inset 0 1px 2px rgba(255,255,255,0.8);
        border: 4px solid rgba(255, 255, 255, 0.9);
        overflow: hidden;
        position: relative;
        background: linear-gradient(135deg, #ffffff, #fff8fa);
        transition: all 0.3s ease;
    }

    .logo-circle:hover {
        transform: translateY(-5px);
        box-shadow: 
            0 12px 40px rgba(255, 105, 180, 0.3),
            0 4px 12px rgba(0,0,0,0.15),
            inset 0 1px 2px rgba(255,255,255,0.9);
    }

    .logo-circle::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(from 0deg, transparent, rgba(255, 105, 180, 0.1), transparent);
        animation: rotate 10s linear infinite;
        z-index: -1;
    }

    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    .logo-circle img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        border-radius: 50%;
    }

    /* === ENHANCED TITLE & SUBTITLE === */
    h1.app-title {
        text-align: center;
        background: linear-gradient(135deg, #7e0000, #c41e3a, #e91e63);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.8px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        position: relative;
    }

    @keyframes twinkle {
        0%, 100% { opacity: 0.5; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.2); }
    }

    p.app-subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        max-width: 450px;
        margin: 0 auto 0rem auto;
        line-height: 1.6;
        font-weight: 500;
        background: rgba(255, 255, 255, 0.6);
        padding: 0.8rem 1.2rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 105, 180, 0.1);
        backdrop-filter: blur(10px);
    }

    /* === ENHANCED FEATURE SECTION === */
    .features-container {
        max-width: 720px;
        margin: 1.5rem auto;
        padding: 0 1rem;
    }

    .feature-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.7), rgba(255, 248, 250, 0.8));
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-radius: 16px;
        padding: 1.2rem;
        margin: 1.6rem 0;
        border: 1px solid rgba(255, 105, 180, 0.15);
        box-shadow: 
            0 8px 25px rgba(0,0,0,0.08),
            0 2px 10px rgba(255, 105, 180, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
        transition: left 0.5s ease;
    }

    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 
            0 12px 35px rgba(0,0,0,0.12),
            0 4px 15px rgba(255, 105, 180, 0.2);
        border-color: rgba(255, 105, 180, 0.3);
    }

    .feature-card:hover::before {
        left: 100%;
    }

    .feature-card h4 {
        margin: 0 0 0.8rem 0;
        font-size: 1.15rem;
        display: flex;
        align-items: center;
        gap: 10px;
        font-weight: 700;
        color: #4a148c;
    }

    .feature-card p {
        margin: 0;
        font-size: 0.95rem;
        line-height: 1.5;
        color: #555;
    }

    /* === ENHANCED TAB STYLING === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        justify-content: center;
        max-width: 420px;
        margin: 0 auto 2rem auto;
        padding: 8px;
        background: rgba(255, 255, 255, 0.4);
        border-radius: 16px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 105, 180, 0.1);
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.6);
        color: #d81b60 !important;
        font-weight: 700;
        font-size: 1.05rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid transparent;
        min-width: 120px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #ff6b9d, #ff8fab) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 4px 15px rgba(255, 107, 157, 0.4);
        transform: translateY(-1px);
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #ffd1e0, #ffb3c6);
        color: white !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 107, 157, 0.3);
    }

    /* === ENHANCED INPUT STYLING WITH PINK UNDERLINES === */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.9) !important;
        border: none !important;
        border-bottom: 2px solid #e0e0e0 !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 0.8rem 1rem !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: none !important;
    }

    .stTextInput > div > div > input:focus {
        border-bottom: 3px solid #ff6b9d !important;
        box-shadow: 
            0 3px 0 0 rgba(255, 107, 157, 0.3),
            0 4px 20px rgba(255, 107, 157, 0.15) !important;
        outline: none !important;
        background: rgba(255, 255, 255, 0.98) !important;
        transform: translateY(-2px);
    }

    .stTextInput > div > div > input::placeholder {
        color: #999 !important;
        font-weight: 400 !important;
    }

    /* === BEAUTIFUL GRADIENT BUTTONS MATCHING YOUR IMAGE === */
    .stButton > button {
        background: linear-gradient(135deg, #ff6b9d 0%, #ff8fab 50%, #ffa8c4 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 
            0 4px 15px rgba(255, 107, 157, 0.4),
            0 2px 8px rgba(255, 107, 157, 0.2) !important;
        position: relative;
        overflow: hidden;
        cursor: pointer;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
        width: 100% !important;
        margin-top: 1.5rem !important;
        height: 50px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.25), transparent);
        transition: left 0.6s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #ff5a8a 0%, #ff7aa0 50%, #ff95b7 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 
            0 8px 25px rgba(255, 107, 157, 0.5),
            0 4px 15px rgba(255, 107, 157, 0.3) !important;
    }

    .stButton > button:hover::before {
        left: 100%;
    }

    .stButton > button:active {
        transform: translateY(0px) !important;
        box-shadow: 
            0 3px 12px rgba(255, 107, 157, 0.4),
            0 1px 6px rgba(255, 107, 157, 0.2) !important;
    }

    /* === ENHANCED CONTAINER LAYOUT === */
    div[data-testid="column"]:nth-child(1) {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.6), rgba(255, 248, 250, 0.7));
        border-radius: 20px;
        padding: 2rem;
        margin-right: 1rem;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 105, 180, 0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.08);
    }

    div[data-testid="column"]:nth-child(2) {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.5), rgba(255, 248, 250, 0.6));
        border-radius: 20px;
        padding: 2rem;
        margin-left: 1rem;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 105, 180, 0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.08);
    }

    .privacy-container {
        background: linear-gradient(135deg, #ffe6f0, #ffb3d1);
        border: 1px solid #e91e63;
        box-shadow: 0 6px 20px rgba(233, 30, 99, 0.3);
    }

    .privacy-container h4 {
        color: #c2185b !important;
        margin-top: 0;
        text-align: center;
    }

    /* === SUCCESS/WARNING MESSAGE STYLING === */
    .stAlert {
        border-radius: 12px !important;
        border: none !important;
        backdrop-filter: blur(10px) !important;
    }

    .stSuccess {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.1), rgba(139, 195, 74, 0.1)) !important;
        border-left: 4px solid #4CAF50 !important;
    }

    .stWarning {
        background: linear-gradient(135deg, rgba(255, 152, 0, 0.1), rgba(255, 193, 7, 0.1)) !important;
        border-left: 4px solid #FF9800 !important;
    }

    .stError {
        background: linear-gradient(135deg, rgba(244, 67, 54, 0.1), rgba(255, 82, 82, 0.1)) !important;
        border-left: 4px solid #f44336 !important;
    }

    /* === RESPONSIVE ENHANCEMENTS === */
    @media (max-width: 768px) {
        div[data-testid="column"]:nth-child(1),
        div[data-testid="column"]:nth-child(2) {
            margin: 0.5rem 0;
            padding: 1.5rem;
        }
        
        .logo-circle {
            width: 160px;
            height: 160px;
        }
        
        h1.app-title {
            font-size: 2rem;
        }
        
        p.app-subtitle {
            font-size: 1rem;
            padding: 0.6rem 1rem;
        }

        .feature-card {
            padding: 1rem;
            margin: 1rem 0;
        }

        .stTabs [data-baseweb="tab-list"] {
            max-width: 100%;
        }
        
        .stButton > button {
            padding: 0.7rem 1.5rem !important;
            font-size: 0.9rem !important;
            height: 45px !important;
        }
    }

    @media (min-width: 1200px) {
        .block-container {
            max-width: 1300px;
            padding: 1rem 2rem;
            margin: auto;
        }
    }
    </style>
    """, unsafe_allow_html=True)
        
    # 🖼️ Two-Column Layout
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        # Logo + Title + Subtitle
        st.markdown("""
        <div style="text-align: center;">
            <div class="logo-circle">
                <img src="https://i.imgur.com/SjsKElM.png" alt="App Logo">
            </div>
            <h1 class="app-title">Menstrual Health & PCOS Risk Detection App</h1>
            <p class="app-subtitle">Your journey to cycle harmony & self-care starts here</p>
        </div>
        """, unsafe_allow_html=True)

        # Tabs for Login/Register
        tab1, tab2 = st.tabs(["Login", "Create Account"])

        with tab1:
            # --- ERROR DISPLAY AT VERY TOP ---
            login_error = None
            
            username = st.text_input("Username Label", placeholder="Username", key="login_username", label_visibility="collapsed")
            password = st.text_input("Password Label", type="password", placeholder="Password", key="login_password", label_visibility="collapsed")

            if st.button("Begin Your Journey", key="login_button", type="primary"):
                if username and password:
                    # Check if user exists first
                    if not check_user_exists(username, None):
                        login_error = "❌ User does not exist. Please register first."
                    else:
                        # Verify credentials
                        is_valid, user_id = verify_user(username, password)
                        if is_valid:
                            st.session_state.logged_in = True
                            st.session_state.username = username
                            st.session_state.user_id = user_id
                            st.session_state["scroll_to_top_intro"] = True
                            st.success("Welcome back!")
                            st.rerun()
                        else:
                            login_error = "❌ Invalid password. Please try again."
                else:
                    login_error = "⚠️ Please enter both username and password"
            
            # --- SHOW WELCOME OR ERROR AT TOP ---
            if login_error:
                st.error(login_error)
            else:
                st.success("🎉 Welcome back!")

        with tab2:
            # --- ERROR DISPLAY AT VERY TOP ---
            reg_error = None
            
            new_username = st.text_input("New Username Label", placeholder="Choose a username", key="reg_username", label_visibility="collapsed")
            new_email = st.text_input("New Email Label", placeholder="Your email", key="reg_email", label_visibility="collapsed")
            new_password = st.text_input("New Password Label", type="password", placeholder="Create a password", key="reg_password", label_visibility="collapsed")

            if st.button("Join", key="register_button", type="primary"):
                if new_username and new_email and new_password:
                    # 1. Check email format
                    if not is_valid_email(new_email):
                        reg_error = "❌ Please enter a valid email address (e.g., user@example.com)"
                    
                    # 2. Check if username/email already exists
                    elif check_user_exists(new_username, new_email):
                        reg_error = "❌ Username or email already exists. Please choose different ones."
                    
                    # 3. If all validations pass, create user
                    else:
                        new_user_id = create_user(new_username, new_email, new_password)
                        if new_user_id:
                            st.session_state.logged_in = True
                            st.session_state.username = new_username
                            st.session_state.user_id = new_user_id
                            st.session_state["scroll_to_top_intro"] = True
                            st.success("🎉 Welcome! Your self-care journey begins now.")
                            st.rerun()
                        else:
                            reg_error = "❌ Registration failed. Please try again."
                else:
                    reg_error = "⚠️ Please fill in all fields 💕"
            
            # --- SHOW WELCOME OR ERROR AT TOP ---
            if reg_error:
                st.error(reg_error)
            else:
                st.success("🎉 Welcome! Your self-care journey begins now.")

    with col2:
        # Features Section
        st.markdown("""
        <div class="features-container">
            <div class="feature-card">
                <h4>📽️ PCOS Video Education</h4>
                <p>Watch engaging videos to understand PCOS and its impact on your health.</p>
            </div>
            <div class="feature-card">
                <h4>🧬 Hormone Insights</h4>
                <p>Learn about your hormones and how they influence your cycle and well-being.</p>
            </div>
            <div class="feature-card">
                <h4>🌸 Phase-Specific Recommendations</h4>
                <p>Get tailored tips for each phase of your menstrual cycle to optimize your health.</p>
            </div>
            <div class="feature-card">
                <h4>🔍 PCOS Risk Detection</h4>
                <p>Assess your PCOS risk (low, medium, high). If high risk is detected, complete an additional questionnaire on page 2 to refine your analysis.</p>
            </div>
            <div class="feature-card privacy-container">
                <h4>Your Privacy Matters</h4>
                <p style="font-size:1rem; text-align:center; color:#555;">
                    <strong>We use your data to personalize insights and improve Cyla.</strong>
                </p>
                <ul style="padding-left:1.2rem; color:#555; font-size:0.95rem;">
                    <li>Your data helps us provide accurate cycle tracking and PCOS risk analysis.</li>
                    <li>We <strong>do not share or sell</strong> your information to third parties.</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
# Page 1
def page_introduction():
    st.markdown("""
    <style>
    /* === ENHANCED BACKGROUND === */
    .stApp {
        background: linear-gradient(135deg, #ffffff 0%, #fff8fa 20%, #ffeef5 40%, #fff0f6 60%, #ffeef2 80%, #ffffff 100%);
        background-attachment: fixed;
        min-height: 100vh;
    }
    .stButton > button {
    background: linear-gradient(135deg, #ff6b9d, #ff8fab, #ffa8c4) !important;
    color: white !important;
    border: none !important;
    border-radius: 15px !important;
    padding: 0.8rem 2rem !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 6px 20px rgba(255, 107, 157, 0.3) !important;
    position: relative;
    overflow: hidden;
    cursor: pointer;
    text-transform: none !important;
    letter-spacing: 0.5px;
    }
    /* === FLOATING BACKGROUND ELEMENTS === */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(circle at 20% 30%, rgba(255, 182, 193, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 70%, rgba(255, 105, 180, 0.08) 0%, transparent 50%),
            radial-gradient(circle at 50% 50%, rgba(255, 192, 203, 0.05) 0%, transparent 70%);
        animation: floatBg 25s ease-in-out infinite;
        z-index: -1;
    }

    @keyframes floatBg {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(1deg); }
    }

    @keyframes sparkle {
        0%, 100% { opacity: 0.6; transform: scale(1) rotate(0deg); }
        50% { opacity: 1; transform: scale(1.1) rotate(5deg); }
    }

    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* === ENHANCED INFO BOXES === */
    .info-box {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(255, 248, 250, 0.95));
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 
            0 8px 32px rgba(255, 105, 180, 0.15),
            0 2px 8px rgba(0, 0, 0, 0.08),
            inset 0 1px 2px rgba(255, 255, 255, 0.9);
        border: 2px solid rgba(255, 105, 180, 0.2);
        animation: fadeInUp 1s ease-out;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }

    .info-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
        transition: left 0.6s ease;
    }

    .info-box:hover {
        transform: translateY(-5px);
        box-shadow: 
            0 15px 40px rgba(255, 105, 180, 0.2),
            0 5px 15px rgba(0, 0, 0, 0.1),
            inset 0 1px 2px rgba(255, 255, 255, 1);
        border-color: rgba(255, 105, 180, 0.3);
    }

    .info-box:hover::before {
        left: 100%;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* === ENHANCED TABS === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        justify-content: center;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.8), rgba(255, 248, 250, 0.9));
        border-radius: 20px;
        padding: 10px;
        backdrop-filter: blur(15px);
        border: 2px solid rgba(255, 105, 180, 0.15);
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
    }

    .stTabs [data-baseweb="tab"] {
        height: 55px;
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.7);
        color: #d81b60 !important;
        font-weight: 700;
        font-size: 1.1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 2px solid transparent;
        min-width: 140px;
        position: relative;
        overflow: hidden;
    }

    .stTabs [data-baseweb="tab"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.5), transparent);
        transition: left 0.5s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #ff6b9d, #ff8fab, #ffa8c4) !important;
        color: white !important;
        border: 2px solid rgba(255, 255, 255, 0.4);
        box-shadow: 
            0 6px 20px rgba(255, 107, 157, 0.4),
            0 2px 8px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #ffd1e0, #ffb3c6);
        color: white !important;
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(255, 107, 157, 0.3);
        border-color: rgba(255, 105, 180, 0.3);
    }

    .stTabs [data-baseweb="tab"]:hover::before {
        left: 100%;
    }

    /* === ENHANCED HEADERS === */
    h2, h3, h4, h5 {
        color: #4a148c !important;
        font-weight: 700 !important;
    }

    h2 {
        font-size: 2rem !important;
        background: linear-gradient(135deg, #7e0000, #c41e3a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* === ENHANCED IMAGES === */
    .stImageHoverWrapper img {
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        border-radius: 20px !important;
        border: 3px solid rgba(255, 105, 180, 0.3) !important;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }

    .stImageHoverWrapper img:hover {
        transform: scale(1.05) translateY(-5px);
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.2),
            0 8px 25px rgba(255, 105, 180, 0.3);
        border-color: rgba(255, 105, 180, 0.5) !important;
    }

    .stImageHoverWrapper {
        margin: 2rem 0;
    }

    /* === ENHANCED VIDEO CONTAINER === */
    .stVideo {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 12px 30px rgba(0,0,0,0.15);
        border: 3px solid rgba(255, 105, 180, 0.2);
        transition: all 0.3s ease;
    }

    .stVideo:hover {
        transform: translateY(-3px);
        box-shadow: 0 18px 40px rgba(0,0,0,0.2);
        border-color: rgba(255, 105, 180, 0.4);
    }

    /* === ENHANCED TABLE === */
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 2rem 0;
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(255, 248, 250, 0.98));
        backdrop-filter: blur(10px);
    }

    table thead tr {
        background: linear-gradient(135deg, #ff6b9d, #ff8fab) !important;
        color: white !important;
    }

    table th {
        padding: 15px !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        text-align: center !important;
        border: none !important;
    }

    table td {
        padding: 15px !important;
        border: 1px solid rgba(255, 105, 180, 0.15) !important;
        font-size: 0.95rem !important;
        line-height: 1.5 !important;
        transition: all 0.2s ease;
    }

    table tr:nth-child(even) {
        background: rgba(255, 240, 245, 0.5);
    }

    table tr:hover {
        background: rgba(255, 105, 180, 0.1);
        transform: scale(1.01);
    }

    /* === ENHANCED GRADIENT DIVIDERS === */
    .gradient-divider {
        height: 4px;
        background: linear-gradient(90deg, #ff6b9d, #ff8fab, #ffa8c4, #ff8fab, #ff6b9d);
        background-size: 200% 100%;
        animation: gradientShift 3s ease-in-out infinite;
        margin: 2.5rem 0;
        border-radius: 2px;
        box-shadow: 0 2px 8px rgba(255, 107, 157, 0.3);
    }

    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }

    /* === ENHANCED TIP BOXES === */
    .tip-box {
        border-radius: 15px;
        padding: 1.2rem;
        border: 2px solid rgba(255, 182, 193, 0.3);
        background: linear-gradient(135deg, rgba(255, 246, 251, 0.9), rgba(255, 240, 245, 0.95));
        backdrop-filter: blur(10px);
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }

    .tip-box::before {
        content: '💡';
        position: absolute;
        top: -5px;
        right: 15px;
        font-size: 1.5rem;
        animation: bounce 2s ease-in-out infinite;
    }

    @keyframes bounce {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
    }

    .tip-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(255, 182, 193, 0.2);
        border-color: rgba(255, 182, 193, 0.5);
    }

    /* === ENHANCED COLUMNS === */
    div[data-testid="column"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.4), rgba(255, 248, 250, 0.6));
        border-radius: 20px;
        padding: 1.5rem;
        margin: 0.5rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 105, 180, 0.1);
        box-shadow: 0 8px 25px rgba(0,0,0,0.06);
        transition: all 0.3s ease;
    }

    div[data-testid="column"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.1);
        border-color: rgba(255, 105, 180, 0.2);
    }

    /* === ENHANCED EXPANDER === */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.8), rgba(255, 248, 250, 0.9)) !important;
        border-radius: 12px !important;
        border: 2px solid rgba(255, 105, 180, 0.2) !important;
        transition: all 0.3s ease !important;
    }

    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, rgba(255, 240, 245, 0.9), rgba(255, 228, 238, 0.95)) !important;
        border-color: rgba(255, 105, 180, 0.3) !important;
        transform: translateY(-1px);
    }

    .streamlit-expanderContent {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(255, 248, 250, 0.98)) !important;
        border-radius: 0 0 12px 12px !important;
        border: 2px solid rgba(255, 105, 180, 0.15) !important;
        border-top: none !important;
        backdrop-filter: blur(10px) !important;
    }

    /* === RESPONSIVE DESIGN === */
    @media (max-width: 768px) {
        h1 {
            font-size: 2.2rem !important;
        }

        h1::after {
            right: -60px;
            font-size: 1.2rem;
        }

        .info-box {
            padding: 1.5rem;
            margin: 1rem 0;
        }

        .stTabs [data-baseweb="tab"] {
            min-width: 100px;
            font-size: 1rem;
        }

        div[data-testid="column"] {
            margin: 0.25rem;
            padding: 1rem;
        }
    }

    /* === ENHANCED LIST STYLING === */
    .info-box ul {
        list-style: none;
        padding-left: 0;
    }

    .info-box li {
        position: relative;
        padding-left: 25px;
        margin: 8px 0;
        line-height: 1.6;
    }

    .info-box li::before {
        content: '🌸';
        position: absolute;
        left: 0;
        top: 0;
    }

    /* === SOURCE LINK STYLING === */
    .source-link {
        text-align: center;
        font-size: 14px;
        color: #666;
        margin-top: 15px;
        padding: 10px;
        background: rgba(255, 255, 255, 0.7);
        border-radius: 10px;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 105, 180, 0.1);
    }

    .source-link a {
        color: #e91e63;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .source-link a:hover {
        color: #c2185b;
        text-decoration: underline;
    }
    </style>
    """, unsafe_allow_html=True)

    # Check if we need to scroll to top
    if st.session_state.get("scroll_to_top_intro", False):
        scroll_to_here(0, key='top_intro')  # Scroll to the top of the page
        st.session_state.scroll_to_top_intro = False  # Reset the state after scrolling
    st.title("Understanding Your Menstrual Health")

    # Tabs for video and pictures
    tab1, tab2, tab3 = st.tabs(["Video", "Hormones", "Cycle Stages"])

    # Tab 1: Video Section
    with tab1:
        st.header("Menstrual Health Video")
        # Layout: Video on the left, Info on the right
        left, right = st.columns([3, 2])
        # Left column (video)
        with left:
            st.video("https://youtu.be/Mc5iK0AtGNc?si=VAE0ZEc-lAcfo4no")

            # Centered source link below video
            st.markdown(
                """
                <div class="source-link">
                    Hormone Graph by 
                    <a href="https://www.forthwithlife.co.uk/wp-content/uploads/2021/03/hormone-graph.png" 
                    target="_blank" rel="noopener noreferrer">
                        ForthWithLife
                    </a>
                </div>
                """,
                unsafe_allow_html=True
            )
        # Right column (info)
        with right:
            st.subheader("Why watch this short video?")
            st.markdown("""
                    - **Learn how hormones change across the cycle.**  
                    - **Understand the menstrual phases at a glance.**   
                    - **Quick tips for self-care per phase.**

            """, unsafe_allow_html=True)

            # Tip box
            st.markdown(
                f"""
                <div class="tip-box">
                    <strong>Tip:</strong> Watch the video, then try the 'Phase & Risk' page to get personalized recommendations.
                </div>
                """,
                unsafe_allow_html=True
            )

    # Tab 2: Hormones Throughout the Cycle
    with tab2:
        st.header("Hormone Levels Throughout the Cycle")
        # Columns: Image on the left, Info on the right
        left, right = st.columns([3, 2])
        # Left column (image)
        with left:
            image_url = "https://www.forthwithlife.co.uk/wp-content/uploads/2021/03/hormone-graph.png"
            st.markdown(
                f"""
                <div class="stImageHoverWrapper" style="text-align: center;">
                    <img src="{image_url}" width="600" style="border-radius: 15px; border: 3px solid #ff99aa;" />
                    <div class="source-link">
                        Hormone Graph by 
                        <a href="https://www.forthwithlife.co.uk/wp-content/uploads/2021/03/hormone-graph.png" target="_blank" rel="noopener noreferrer">
                            ForthWithLife
                        </a>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Right column (info)
        with right:
            st.write("""
            This chart shows the fluctuation of key hormones such as estrogen, progesterone, and LH (luteinizing hormone) throughout the different phases of your cycle.
            """)

            st.markdown("""
                - **Follicular Phase:** Estrogen rises, helping the follicles in your ovaries develop.  
                - **Ovulation:** Estrogen peaks and LH surges, triggering the release of an egg.  
                - **Luteal Phase:** Progesterone increases to prepare the uterus for a potential pregnancy.
                """, unsafe_allow_html=True)

            st.markdown(
                f"""
                <div class="tip-box">
                    <strong>Tip:</strong> Hormonal fluctuations are normal, but tracking your cycle can help you better understand your body!
                </div>
                """,
                unsafe_allow_html=True
            )

    # Tab 3: Stages of the Menstrual Cycle
    with tab3:
        st.header("Stages of the Menstrual Cycle")

        # Columns: Image on the left, Info on the right
        left, right = st.columns([3, 2])

        # Left column (image)
        with left:
            image_url = "https://img.freepik.com/premium-vector/female-reproductive-system-infographic-stages-menstrual-cycle-vector_980832-896.jpg?w=1380"
            
            st.markdown(
                f"""
                <div class="stImageHoverWrapper" style="text-align: center;">
                    <img src="{image_url}" width="600" style="border-radius: 15px; border: 3px solid #ff99aa;" />
                    <div class="source-link">
                        Cycle Stages Infographic by 
                        <a href="https://www.freepik.com/" target="_blank" rel="noopener noreferrer">
                            Freepik
                        </a>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Right column (info)
        with right:
            st.write("""
            The menstrual cycle consists of several stages, each characterized by hormonal fluctuations and physiological changes in the body. Understanding these stages can help you better track your cycle and recognize any abnormalities.
            """)
            st.write("""
            • **Menstrual Phase:** The shedding of the uterine lining (period).  
            • **Follicular Phase:** Follicles in the ovaries mature, leading up to ovulation.  
            • **Ovulation:** The release of an egg from the ovary, ready for fertilization.  
            • **Luteal Phase:** The uterine lining thickens in preparation for a potential pregnancy.
            """)

            st.markdown(
                f"""
                <div class="tip-box">
                    <strong>Tip:</strong> Knowing each phase of your cycle helps you identify when symptoms like cramps or mood swings might occur!
                </div>
                """,
                unsafe_allow_html=True
            )
            
    # Existing content
    st.markdown("<div class='gradient-divider'></div>", unsafe_allow_html=True)
    st.markdown("## Learn More About Menstrual Health")
    st.markdown("""
    <div class='info-box'>
        <p><em>Your menstrual cycle is a reflection of your body's internal health. By understanding its patterns, you can support fertility, recognize hormonal imbalances, and take proactive control of your wellness.</em></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Understanding the Four Phases of the Menstrual Cycle")
    st.markdown("""
    <div class='info-box'>
        <p>Each phase of the menstrual cycle has its own hormonal profile and physiological role. Learning about them can help you optimize your health and lifestyle choices.</p>
    </div>
    """, unsafe_allow_html=True)

    table_html = """
    <table>
    <thead>
    <tr>
    <th>Phase</th>
    <th>Typical Days</th>
    <th>Description</th>
    <th>Recommendations</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td><strong>Menstrual</strong></td>
    <td>1–5</td>
    <td>Shedding of the uterine lining; a natural reset.</td>
    <td>Rest, stay hydrated, and focus on recovery.</td>
    </tr>
    <tr>
    <td><strong>Follicular</strong></td>
    <td>6–13</td>
    <td>Estrogen rises, preparing the body for ovulation.</td>
    <td>Engage in light physical activity.</td>
    </tr>
    <tr>
    <td><strong>Ovulatory</strong></td>
    <td>14–16</td>
    <td>Egg is released; fertility peaks.</td>
    <td>Track ovulation and consider conception timing.</td>
    </tr>
    <tr>
    <td><strong>Luteal</strong></td>
    <td>17–28</td>
    <td>Progesterone increases, supporting potential pregnancy.</td>
    <td>Reduce bloating with herbal teas and manage stress.</td>
    </tr>
    </tbody>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("<div class='gradient-divider'></div>", unsafe_allow_html=True)

    st.markdown("### An Overview of Polycystic Ovary Syndrome (PCOS)")
    st.markdown("""
    <div class='info-box'>
        <p>Polycystic Ovary Syndrome (PCOS) is a common hormonal disorder affecting approximately 1 in 10 individuals with ovaries. It may involve:</p>
        <ul>
            <li>Irregular or absent periods</li>
            <li>Elevated levels of androgens, causing acne or excess hair growth</li>
            <li>Polycystic ovaries, often visible on ultrasound</li>
        </ul>
        <p>PCOS can also impact metabolic and emotional health, increasing the risk for conditions such as insulin resistance, diabetes, and cardiovascular disease.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("PCOS Symptoms")
    image_url = "https://th.bing.com/th/id/R.75366f3f39d2b32d53306063e42dfffe?rik=4ldw75oikYueVw&riu=http%3a%2f%2fthegoldenlady.net%2fwp-content%2fuploads%2f2022%2f07%2fpcossym.jpeg&ehk=siPTlS%2ftQ%2fO7Bm1K614PAvGRCl0bXDsIxcqzyBY57jA%3d&risl=&pid=ImgRaw&r=0"
    
    st.markdown(
        f"""
        <div class="stImageHoverWrapper" style="text-align: center;">
            <img src="{image_url}" width="600" />
            <div class="source-link">
                PCOS Infographic by 
                <a href="https://thegoldenlady.net/2022/07/pcos-symptoms/" target="_blank" rel="noopener noreferrer">
                    Dr. Becky Campbell
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### What Do Polycystic Ovaries Look Like?")

    st.markdown(
        f"""
        <div class="stImageHoverWrapper" style="text-align: center;">
            <img src="https://images.ctfassets.net/ld5gan8tjh6b/1DTRynouKgzWzaLMLwiHLG/b183fbfd3394df0945c3e975ffb06d9e/ultrasound_of_normal_vs_PCOS_ovary.png" width="600" />
            <div class="source-link">
                Image by 
                <a href="https://healthmatch.io/" target="_blank" rel="noopener noreferrer">
                    HealthMatch
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Additional Information")
    with st.container():
        st.markdown(
            """
            <div class='info-box'>
                <h5>Estrogen</h5>
                <p>Supports uterine lining growth, regulates your cycle, and influences bone density and mood. 
                Imbalances can lead to irregular periods or mood changes.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with st.container():
        st.markdown("""
        <div class='info-box'>
            <h5>Progesterone</h5>
            <p>Produced after ovulation, it prepares the uterus for a potential pregnancy. Low levels can result in spotting or shorter cycles.</p>
        </div>
        """, unsafe_allow_html=True)

    with st.container():
        st.markdown("""
        <div class='info-box'>
            <h5>Common Myths About Menstrual Health</h5>
            <ul>
                <li><strong>"A 28-day cycle is the only normal"</strong> – In reality, 21–35 days is considered a healthy range.</li>
                <li><strong>"Severe pain is just part of menstruation"</strong> – While mild discomfort is normal, intense pain should be evaluated.</li>
                <li><strong>"You can't get pregnant during your period"</strong> – Though rare, it is still possible, especially with irregular cycles.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
        <h5>Tips for Managing PCOS</h5>
        <ul>
            <li>Follow a balanced, low-glycemic diet rich in fiber</li>
            <li>Engage in regular physical activity</li>
            <li>Incorporate stress management practices like yoga or meditation</li>
            <li>Consult a healthcare provider regarding treatment options or supplements</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 💡 How to Use This App")

    with st.expander("Getting Started"):
        st.markdown("""
            **Page 2 – Phase Identification & PCOS Screening**

            On the **"Phase & Risk"** page, you'll:
            - Discover your current menstrual phase with personalized recommendations 🌙
            - Complete a quick symptom assessment to estimate your PCOS risk level 🔍
            - Receive immediate insights about your hormonal health 💫

            ---
            **What happens if you're at higher risk?** 🤔
            **Page 3 - Detailed Analysis?**

            If our screening suggests elevated PCOS risk, you'll unlock access to our **detailed analysis** - a comprehensive assessment that dives deeper into your health profile with more precise evaluation.

            *Curious about what the detailed analysis includes?*
            - Advanced hormone tracking and interpretation
            - Personalized lifestyle recommendations
            - Deeper insight into your unique hormonal patterns
            - Actionable steps tailored just for you

            ---

            **Why This Matters** ✨

            Your hormonal health influences everything from mood and energy to fertility and long-term wellness.  
            This app helps you **understand your body's signals** and provides personalized guidance - giving you clarity and confidence in your health journey.
            """)

def get_detailed_history(user_id):
    conn = get_db_connection()
    if not conn:
        return []
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM detailed_inputs WHERE user_id=%s ORDER BY submitted_at DESC", (user_id,))
            return cursor.fetchall()
    except Exception as e:
        st.error(f"Error retrieving history: {e}")
        return []
    finally:
        conn.close()

# Handle navigation flags (must be before any widgets to avoid StreamlitAPIException)
if st.session_state.get("switch_to_detailed", False):
    st.session_state["nav_radio"] = "Detailed Analysis"
    del st.session_state["switch_to_detailed"]

# Page 2 - Phase & Risk Analysis
def page_phase_risk():
    st.session_state.setdefault("risk_category_page2", None)
    st.session_state.setdefault("recommendations_page2", "")
    st.session_state.setdefault("symptom_data", {})
    st.session_state.setdefault("symptom_form_submitted_page2", False)
    st.session_state.setdefault("lmp_phase_checked", False)
    st.session_state.setdefault("lmp_phase", None)
    st.session_state.setdefault("lmp_date", None)
    st.session_state.setdefault("already_redirected", False)

    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #ffffff 0%, #fff8fa 20%, #ffeef5 40%, #fff0f6 60%, #ffeef2 80%, #ffffff 100%);
        background-attachment: fixed;
        min-height: 100vh;
    }
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(circle at 20% 30%, rgba(255, 182, 193, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 70%, rgba(255, 105, 180, 0.08) 0%, transparent 50%),
            radial-gradient(circle at 50% 50%, rgba(255, 192, 203, 0.05) 0%, transparent 70%);
        animation: floatBg 25s ease-in-out infinite;
        z-index: -1;
    }
    @keyframes floatBg {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(1deg); }
    }
    .section-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(255, 248, 250, 0.98));
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 25px;
        padding: 2.5rem;
        margin: 2rem 0;
        box-shadow: 
            0 15px 35px rgba(255, 105, 180, 0.15),
            0 5px 15px rgba(0, 0, 0, 0.08),
            inset 0 1px 3px rgba(255, 255, 255, 0.9);
        border: 2px solid rgba(255, 105, 180, 0.2);
        animation: fadeInUp 1s ease-out;
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .section-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255, 105, 180, 0.03) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
        z-index: -1;
    }
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .section-card:hover {
        transform: translateY(-8px);
        box-shadow: 
            0 25px 50px rgba(255, 105, 180, 0.25),
            0 10px 25px rgba(0, 0, 0, 0.15),
            inset 0 1px 3px rgba(255, 255, 255, 1);
        border-color: rgba(255, 105, 180, 0.3);
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .page-title {
        font-size: 3.2rem !important;
        background: linear-gradient(135deg, #7e0000, #c41e3a, #e91e63, #ff69b4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        font-weight: 800;
        letter-spacing: -1px;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        animation: fadeInDown 1.2s ease-out;
        margin-bottom: 2rem !important;
        position: relative;
    }
    .section-title {
        font-size: 2.5rem !important;
        background: linear-gradient(135deg, #7e0000, #c41e3a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        font-weight: 700;
        margin-bottom: 1rem !important;
        position: relative;
    }
    .section-subtitle {
        color: #2c3e50 !important;
        font-size: 1.1rem !important;
        text-align: center;
        opacity: 0.9;
        font-weight: 400;
        line-height: 1.6;
    }
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stForm {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.8), rgba(255, 248, 250, 0.9));
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 1rem;
        border: 2px solid rgba(255, 105, 180, 0.2);
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    .stForm:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.12);
        border-color: rgba(255, 105, 180, 0.3);
    }
    .stRadio > div {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(255, 248, 250, 0.95));
        border-radius: 15px;
        padding: 0.5rem;
        border: 2px solid rgba(255, 105, 180, 0.15);
        margin: 0.3rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    .stRadio > div:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(255, 105, 180, 0.2);
        border-color: rgba(255, 105, 180, 0.3);
        background: linear-gradient(135deg, rgba(255, 240, 245, 0.9), rgba(255, 228, 238, 0.95));
    }
    .stRadio label {
        font-weight: 600 !important;
        color: #2c3e50 !important;
        font-size: 1rem !important;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .stNumberInput > div > div > input {
        background-color: #fff !important;
        border-radius: 12px !important;
        padding: 0.6rem !important;
        border: 2px solid #ffd1d9 !important;
        transition: all 0.3s ease !important;
        font-weight: 500 !important;
        color: #8b0026 !important;
        box-shadow: 0 0 10px rgba(255, 153, 170, 0.2) !important;
    }
    .stNumberInput > div > div > input:hover {
        border-color: #ff99aa !important;
        box-shadow: 0 0 10px rgba(255, 153, 170, 0.4) !important;
    }
    .stNumberInput > div > div > input:focus {
        border-color: #ff6680 !important;
        box-shadow: 0 0 10px rgba(255, 153, 170, 0.6) !important;
    }
    .stDateInput > div > div > input {
        background-color: #fff !important;
        border-radius: 12px !important;
        padding: 0.6rem !important;
        border: 2px solid #ffd1d9 !important;
        transition: all 0.3s ease !important;
        font-weight: 500 !important;
        color: #8b0026 !important;
        box-shadow: 0 0 10px rgba(255, 153, 170, 0.2) !important;
    }
    .stDateInput > div > div > input:hover {
        border-color: #ff99aa !important;
        box-shadow: 0 0 10px rgba(255, 153, 170, 0.4) !important;
    }
    .stDateInput > div > div > input:focus {
        border-color: #ff6680 !important;
        box-shadow: 0 0 10px rgba(255, 153, 170, 0.6) !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #ff6b9d, #ff8fab, #ffa8c4) !important;
        color: white !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 0.8rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 6px 20px rgba(255, 107, 157, 0.3) !important;
        position: relative;
        overflow: hidden;
        cursor: pointer;
        text-transform: none !important;
        letter-spacing: 0.5px;
    }
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        background: rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #ff5a8a, #ff7aa0, #ff95b7) !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 30px rgba(255, 107, 157, 0.4) !important;
    }
    .stButton > button:hover::before {
        width: 300px;
        height: 300px;
    }
    .stButton > button:active {
        transform: translateY(-1px) !important;
    }
    .stForm .stButton > button {
        width: 100%;
        margin-top: 1rem;
        background: linear-gradient(135deg, #e91e63, #ad1457, #880e4f) !important;
        font-size: 1.2rem !important;
        padding: 1rem 2rem !important;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 6px 20px rgba(255, 107, 157, 0.3); }
        50% { box-shadow: 0 8px 25px rgba(255, 107, 157, 0.5); }
        100% { box-shadow: 0 6px 20px rgba(255, 107, 157, 0.3); }
    }
    .low-risk {
        background: linear-gradient(135deg, #d4edda, #c3e6cb) !important;
        border: 2px solid #28a745 !important;
        color: #155724 !important;
        animation: successGlow 2s ease-in-out infinite;
    }
    .moderate-risk {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7) !important;
        border: 2px solid #ffc107 !important;
        color: #856404 !important;
        animation: warningGlow 2s ease-in-out infinite;
    }
    .high-risk {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb) !important;
        border: 2px solid #dc3545 !important;
        color: #721c24 !important;
        animation: dangerGlow 2s ease-in-out infinite;
    }
    @keyframes successGlow {
        0%, 100% { box-shadow: 0 0 20px rgba(40, 167, 69, 0.3); }
        50% { box-shadow: 0 0 30px rgba(40, 167, 69, 0.5); }
    }
    @keyframes warningGlow {
        0%, 100% { box-shadow: 0 0 20px rgba(255, 193, 7, 0.3); }
        50% { box-shadow: 0 0 30px rgba(255, 193, 7, 0.5); }
    }
    @keyframes dangerGlow {
        0%, 100% { box-shadow: 0 0 20px rgba(220, 53, 69, 0.3); }
        50% { box-shadow: 0 0 30px rgba(220, 53, 69, 0.5); }
    }
    .phase-display {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(255, 248, 250, 0.98));
        backdrop-filter: blur(20px);
        border-radius: 25px;
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
        border: 2px solid rgba(255, 105, 180, 0.3);
        box-shadow: 0 15px 35px rgba(255, 105, 180, 0.2);
        animation: phaseReveal 1.5s ease-out;
        position: relative;
        overflow: hidden;
    }
    .phase-display::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        animation: shimmer 3s ease-in-out infinite;
    }
    @keyframes phaseReveal {
        from { opacity: 0; transform: scale(0.9) translateY(20px); }
        to { opacity: 1; transform: scale(1) translateY(0); }
    }
    @keyframes shimmer {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    .phase-title {
        font-size: 2.5rem !important;
        font-weight: 800;
        margin-bottom: 1rem;
        color: #000000 !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        background: none !important;
        background-color: transparent !important;
        -webkit-background-clip: unset !important;
        -webkit-text-fill-color: #000000 !important;
        background-clip: unset !important;
        box-shadow: none !important;
    }
    .phase-description {
        font-size: 1.2rem;
        color: #000000 !important;
        line-height: 1.6;
        margin-bottom: 1.5rem;
    }
    div[data-testid="column"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.6), rgba(255, 248, 250, 0.8));
        border-radius: 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 105, 180, 0.15);
        box-shadow: 0 8px 25px rgba(0,0,0,0.06);
        transition: all 0.3s ease;
    }
    div[data-testid="column"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.1);
        border-color: rgba(255, 105, 180, 0.25);
    }
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.8), rgba(255, 248, 250, 0.9)) !important;
        border-radius: 12px !important;
        border: 2px solid rgba(255, 105, 180, 0.2) !important;
        transition: all 0.3s ease !important;
        font-weight: 600 !important;
    }
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, rgba(255, 240, 245, 0.9), rgba(255, 228, 238, 0.95)) !important;
        border-color: rgba(255, 105, 180, 0.3) !important;
        transform: translateY(-1px);
    }
    .streamlit-expanderContent {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(255, 248, 250, 0.98)) !important;
        border-radius: 0 0 12px 12px !important;
        border: 2px solid rgba(255, 105, 180, 0.15) !important;
        border-top: none !important;
        backdrop-filter: blur(10px) !important;
        padding: 1.5rem !important;
    }
    .question-label {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: #2c3e50 !important;
        margin-bottom: 0.5rem !important;
        display: block;
        text-align: left;
    }
    .login-warning {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(45deg, #ffeaa7, #fdcb6e);
        border-radius: 20px;
        margin: 2rem 0;
        border: 2px solid rgba(253, 203, 110, 0.5);
        box-shadow: 0 8px 25px rgba(253, 203, 110, 0.3);
        animation: bounce 2s ease-in-out infinite;
    }
    @keyframes bounce {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
    }
    .symptom-questions-container {
        background: linear-gradient(135deg, #fff0f3, #ffe4e8);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 105, 180, 0.3);
        transition: all 0.3s ease;
    }
    .symptom-questions-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }
    .question-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 105, 180, 0.2);
        box-shadow: 0 4px 15px rgba(255, 105, 180, 0.1);
        transition: all 0.3s ease;
    }
    .question-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 105, 180, 0.2);
    }
    .question-card .stRadio > div {
        background: linear-gradient(135deg, #fff8fa, #ffeef5);
        border-radius: 8px;
        padding: 0.5rem;
        border: 1px solid rgba(255, 105, 180, 0.1);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        display: flex;
        justify-content: center;
        gap: 1rem;
    }
    .question-card .stRadio > div:hover {
        border-color: #ff99aa;
        box-shadow: 0 4px 12px rgba(255, 153, 170, 0.3);
    }
    .question-card .stRadio label {
        font-weight: 600 !important;
        color: #8b0026 !important;
        font-size: 0.95rem !important;
    }
    .question-title {
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        color: #7e0000 !important;
        margin-bottom: 0.5rem;
    }
    .question-icon {
        font-size: 1.2rem;
        color: #ff6680 !important;
    }
    .number-input-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 105, 180, 0.2);
        box-shadow: 0 4px 15px rgba(255, 105, 180, 0.1);
        text-align: center;
        transition: all 0.3s ease;
    }
    .number-input-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 105, 180, 0.2);
    }
    .number-input-card .stNumberInput > div > div > input {
        background: #fff !important;
        border: 2px solid #ffd1d9 !important;
        border-radius: 12px !important;
        padding: 0.6rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        text-align: center !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
        transition: all 0.3s ease !important;
    }
    .number-input-card .stNumberInput > div > div > input:hover {
        border-color: #ff99aa !important;
        box-shadow: 0 0 10px rgba(255, 153, 170, 0.4) !important;
    }
    .number-input-card .stNumberInput > div > div > input:focus {
        border-color: #ff6680 !important;
        box-shadow: 0 0 10px rgba(255, 153, 170, 0.6) !important;
    }
    .email-section {
        background: linear-gradient(135deg, #fff0f3, #ffe4e8);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 105, 180, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .email-section:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }
    .email-section .stRadio > div {
        background: linear-gradient(135deg, #fff8fa, #ffeef5);
        border-radius: 8px;
        padding: 0.5rem;
        border: 1px solid rgba(255, 105, 180, 0.1);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        display: flex;
        justify-content: center;
        gap: 1rem;
    }
    .email-section .stRadio > div:hover {
        border-color: #ff99aa;
        box-shadow: 0 4px 12px rgba(255, 153, 170, 0.3);
    }
    .email-section .stTextInput > div > div > input {
        background: #fff !important;
        border: 2px solid #ffd1d9 !important;
        border-radius: 12px !important;
        padding: 0.6rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
        transition: all 0.3s ease !important;
        color: #8b0026 !important;
    }
    .email-section .stTextInput > div > div > input:hover {
        border-color: #ff99aa !important;
        box-shadow: 0 0 10px rgba(255, 153, 170, 0.4) !important;
    }
    .email-section .stTextInput > div > div > input:focus {
        border-color: #ff6680 !important;
        box-shadow: 0 0 10px rgba(255, 153, 170, 0.6) !important;
    }
    .email-section .stButton > button {
        background: linear-gradient(135deg, #ff6b9d, #ff8fab) !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        padding: 0.6rem 1.5rem !important;
        font-size: 1rem !important;
        box-shadow: 0 4px 15px rgba(255, 107, 157, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    .email-section .stButton > button:hover {
        background: linear-gradient(135deg, #ff5a8a, #ff7aa0) !important;
        box-shadow: 0 6px 20px rgba(255, 107, 157, 0.4) !important;
        transform: translateY(-2px);
    }
    .question-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin: 1rem 0;
    }
    @media (max-width: 768px) {
        .question-grid {
            grid-template-columns: 1fr;
            gap: 0.5rem;
        }
    }
    @media (max-width: 768px) {
        .symptom-questions-container {
            padding: 1rem;
            margin: 0.5rem 0;
        }
        .question-card {
            padding: 0.8rem;
        }
        .question-title {
            font-size: 1rem !important;
        }
        .analyze-button-container .stButton > button {
            padding: 0.8rem 1.5rem !important;
            font-size: 1.1rem !important;
            min-width: 180px;
        }
        .page-title {
            font-size: 2.2rem !important;
        }
        .section-title {
            font-size: 2rem !important;
        }
        .section-card {
            padding: 1.5rem;
            margin: 0.5rem 0;
        }
        div[data-testid="column"] {
            padding: 0.8rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <h1 style='text-align:center;'>
        Phase-Based Recommendations &<br>
        Initial PCOS Screening
    </h1>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-card">
        <h2 class="section-title">Discover Your Menstrual Phase</h2>
        <p class="section-subtitle">
           Unlock personalized wellness insights based on where you are in your cycle (Optional)                
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("lmp_form"):
        st.markdown("<p class='question-label'>When did your last period start?</p>", unsafe_allow_html=True)
        today = datetime.today().date()
        lmp_date = st.date_input(
            "Last Menstrual Period",
            value=today,           # This sets the default to today
            max_value=today,
            label_visibility="collapsed"
        )
        col1, col2, col3 = st.columns([1, 4, 1])
        with col1:
            check_phase_btn = st.form_submit_button("Reveal My Phase")
            if check_phase_btn:
                st.session_state["lmp_date"] = lmp_date
                phase = get_menstrual_phase(lmp_date, today)
                st.session_state["lmp_phase"] = phase
                st.session_state["lmp_phase_checked"] = True
                st.rerun()

    if st.session_state.get("lmp_phase_checked"):
        phase_data = get_phase_recommendation(st.session_state["lmp_phase"])
        display_enhanced_recommendations(phase_data)

    st.markdown("""
    <div class="section-card">
        <h2 class="section-title">PCOS Symptom Check</h2>
        <p class="section-subtitle">
            Quick assessment to check your PCOS risk level
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("symptom_form", clear_on_submit=False):
        st.markdown("""
        <div class="symptom-questions-container">
            <div style="text-align: center; margin-bottom: 1rem;">
                <h3 style="color: #7e0000; font-size: 1.4rem; font-weight: 600; margin: 0;">
                    Tell us about your symptoms
                </h3>
                <p style="color: #8b0026; font-size: 0.95rem; margin: 0.3rem 0 0 0;">
                    Your responses help us provide personalized insights
                </p>
            </div>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            st.markdown("""
            <div class="question-card">
                <div class="question-title">
                    <span class="question-icon">💇‍♀️</span>
                    Excessive hair growth?
                </div>
            </div>
            """, unsafe_allow_html=True)
            hair_growth = st.radio(
                label="",
                options=["No", "Yes"],
                key="hair_growth_input_page2",
                horizontal=True,
                index=["No", "Yes"].index(
                    st.session_state["symptom_data"].get("hair growth(Y/N)", "No")
                ),
                label_visibility="collapsed"
            )
            st.markdown("""
            <div class="question-card">
                <div class="question-title">
                    <span class="question-icon">⚖️</span>
                    Unexpected weight gain?
                </div>
            </div>
            """, unsafe_allow_html=True)
            weight_gain = st.radio(
                label="",
                options=["No", "Yes"],
                key="weight_gain_input_page2",
                horizontal=True,
                index=["No", "Yes"].index(
                    st.session_state["symptom_data"].get("Weight gain(Y/N)", "No")
                ),
                label_visibility="collapsed"
            )
        with col2:
            st.markdown("""
            <div class="question-card">
                <div class="question-title">
                    <span class="question-icon">🌑</span>
                    Skin darkening (acanthosis)?
                </div>
            </div>
            """, unsafe_allow_html=True)
            skin_darkening = st.radio(
                label="",
                options=["No", "Yes"],
                key="skin_darkening_input_page2",
                horizontal=True,
                index=["No", "Yes"].index(
                    st.session_state["symptom_data"].get("Skin darkening (Y/N)", "No")
                ),
                label_visibility="collapsed"
            )
            st.markdown("""
            <div class="question-card">
                <div class="question-title">
                    <span class="question-icon">🍔</span>
                    Frequent fast food consumption?
                </div>
            </div>
            """, unsafe_allow_html=True)
            fast_food = st.radio(
                label="",
                options=["No", "Yes"],
                key="fast_food_input_page2",
                horizontal=True,
                index=["No", "Yes"].index(
                    st.session_state["symptom_data"].get("Fast food (Y/N)", "No")
                ),
                label_visibility="collapsed"
            )
        st.markdown("""
        <div style="margin: 1rem 0;">
            <div class="number-input-card">
                <div class="question-title" style="justify-content: center;">
                    <span class="question-icon">📅</span>
                    What's your cycle length?
                </div>
                <p style="color: #8b0026; font-size: 0.9rem; margin-bottom: 0.5rem;">
                    Enter the number of days in your menstrual cycle (typically 21-35 days)
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        cycle_length = st.number_input(
            label="cycle_length_label",
            min_value=15,
            max_value=60,
            value=st.session_state["symptom_data"].get("Cycle length(days)", 28),
            key="cycle_length_input_page2",
            label_visibility="collapsed"
        )

        st.markdown("""
        <div class="analyze-button-container">
        """, unsafe_allow_html=True)
        submit_btn = st.form_submit_button("Analyze My Risk Level")
        st.markdown("</div>", unsafe_allow_html=True)
        if submit_btn:
            is_irregular_cycle = (cycle_length < 21 or cycle_length > 35)
            yes_count = sum([
                hair_growth == "Yes",
                weight_gain == "Yes",
                skin_darkening == "Yes",
                fast_food == "Yes",
                is_irregular_cycle
            ])
            symptom_data = {
                "hair growth(Y/N)": hair_growth,
                "Weight gain(Y/N)": weight_gain,
                "Skin darkening (Y/N)": skin_darkening,
                "Fast food (Y/N)": fast_food,
                "Cycle length(days)": cycle_length
            }
            if yes_count <= 1:
                risk_category = "Low"
                recommendations = """
                <div class="phase-display low-risk" style="margin: 2rem 0;">
                    <h3 class="phase-title" style="color: #000000;">Low Risk - Amazing!</h3>
                    <p class="phase-description">You're doing fantastic! Keep up the great work!</p>
                </div>
                <div class="section-card">
                    <h3 style="color: #7e0000; text-align: center; margin-bottom: 1.5rem;">Your Personalized Recommendations</h3>
                    <ul style="color: #000000; font-size: 1.1rem; line-height: 1.8; list-style: none; padding-left: 0;">
                        <li style="margin: 1rem 0; padding-left: 2rem; position: relative;">
                            <span style="position: absolute; left: 0; color: #28a745; font-size: 1.2rem;">💖</span>
                            Continue monitoring your menstrual cycle and symptoms for any changes.
                        </li>
                        <li style="margin: 1rem 0; padding-left: 2rem; position: relative;">
                            <span style="position: absolute; left: 0; color: #28a745; font-size: 1.2rem;">🥗</span>
                            Maintain a balanced diet rich in whole foods and regular physical activity.
                        </li>
                        <li style="margin: 1rem 0; padding-left: 2rem; position: relative;">
                            <span style="position: absolute; left: 0; color: #28a745; font-size: 1.2rem;">📱</span>
                            Track your cycle (e.g., with an app) to catch irregularities early.
                        </li>
                        <li style="margin: 1rem 0; padding-left: 2rem; position: relative;">
                            <span style="position: absolute; left: 0; color: #28a745; font-size: 1.2rem;">👩‍⚕️</span>
                            Consult a healthcare provider if new symptoms like irregular periods or hair growth appear.
                        </li>
                        <li style="margin: 1rem 0; padding-left: 2rem; position: relative;">
                            <span style="position: absolute; left: 0; color: #28a745; font-size: 1.2rem;">🔄</span>
                            Re-assess later if symptoms change or worsen.
                        </li>
                    </ul>
                    <p style="color: #6c757d; font-style: italic; text-align: center; margin-top: 1.5rem; font-size: 0.95rem;">
                        💡 Note: These are general tips, not a substitute for medical advice.
                    </p>
                </div>
                """
            elif yes_count == 2:
                risk_category = "Moderate"
                recommendations = """
                <div class="phase-display moderate-risk" style="margin: 2rem 0;">
                    <h3 class="phase-title" style="color: #000000;">⚠️ Moderate Risk - Time to Check In ⚠️</h3>
                    <p class="phase-description">You're doing great, but let's keep an eye on things, okay?</p>
                </div>
                <div class="section-card">
                    <h3 style="color: #7e0000; text-align: center; margin-bottom: 1.5rem;">Your Personalized Recommendations</h3>
                    <ul style="color: #000000; font-size: 1.1rem; line-height: 1.8; list-style: none; padding-left: 0;">
                        <li style="margin: 1rem 0; padding-left: 2rem; position: relative;">
                            <span style="position: absolute; left: 0; color: #ffc107; font-size: 1.2rem;">👩‍⚕️</span>
                            Schedule a chat with your healthcare provider about your symptoms and cycle.
                        </li>
                        <li style="margin: 1rem 0; padding-left: 2rem; position: relative;">
                            <span style="position: absolute; left: 0; color: #ffc107; font-size: 1.2rem;">📓</span>
                            Keep a symptom diary (frequency, severity) to share with your doctor.
                        </li>
                        <li style="margin: 1rem 0; padding-left: 2rem; position: relative;">
                            <span style="position: absolute; left: 0; color: #ffc107; font-size: 1.2rem;">🥗</span>
                            Consider lifestyle tweaks: more whole foods, regular exercise, and stress management.
                        </li>
                        <li style="margin: 1rem 0; padding-left: 2rem; position: relative;">
                            <span style="position: absolute; left: 0; color: #ffc107; font-size: 1.2rem;">🩺</span>
                            Ask about blood tests (e.g., hormone levels) to rule out PCOS or other issues.
                        </li>
                        <li style="margin: 1rem 0; padding-left: 2rem; position: relative;">
                            <span style="position: absolute; left: 0; color: #ffc107; font-size: 1.2rem;">🔄</span>
                            Re-assess later if symptoms change or worsen.
                        </li>
                    </ul>
                    <p style="color: #6c757d; font-style: italic; text-align: center; margin-top: 1.5rem; font-size: 0.95rem;">
                        💡 Note: These are general tips, not a substitute for medical advice.
                    </p>
                </div>
                """
            else:
                risk_category = "High"
                recommendations = """
                <div class="phase-display high-risk" style="margin: 2rem 0;">
                    <h3 class="phase-title" style="color: #000000;">🚨 High Risk - Switched to detailed analysis 🚨</h3>
                    <p class="phase-description">You were redirected because of the risk level, but don't worry, here are some recommendations</p>
                </div>
                <div class="section-card">
                    <h3 style="color: #7e0000; text-align: center; margin-bottom: 1.5rem;">Your Personalized Recommendations</h3>
                    <ul style="color: #000000; font-size: 1.1rem; line-height: 1.8; list-style: none; padding-left: 0;">
                        <li style="margin: 1rem 0; padding-left: 2rem; position: relative;">
                            <span style="position: absolute; left: 0; color: #dc3545; font-size: 1.2rem;">👩‍⚕️</span>
                            Seek medical advice ASAP to discuss possible PCOS or related conditions.
                        </li>
                        <li style="margin: 1rem 0; padding-left: 2rem; position: relative;">
                            <span style="position: absolute; left: 0; color: #dc3545; font-size: 1.2rem;">🩺</span>
                            Ask your doctor about tests: hormone levels (e.g., LH, FSH), ultrasound, or blood sugar.
                        </li>
                        <li style="margin: 1rem 0; padding-left: 2rem; position: relative;">
                            <span style="position: absolute; left: 0; color: #dc3545; font-size: 1.2rem;">💡</span>
                            Early action can manage symptoms and lower risks like insulin resistance or infertility.
                        </li>
                        <li style="margin: 1rem 0; padding-left: 2rem; position: relative;">
                            <span style="position: absolute; left: 0; color: #dc3545; font-size: 1.2rem;">📓</span>
                            Start a symptom log to help your doctor assess your health.
                        </li>
                    </ul>
                    <p style="color: #6c757d; font-style: italic; text-align: center; margin-top: 1.5rem; font-size: 0.95rem;">
                        💡 Note: These are general tips, not a substitute for medical advice.
                    </p>
                </div>
                """
            st.session_state["risk_category_page2"] = risk_category
            st.session_state["recommendations_page2"] = recommendations
            st.session_state["symptom_data"] = symptom_data
            st.session_state["symptom_form_submitted_page2"] = True
            _ = save_symptom_input_to_db(
                st.session_state["user_id"],
                hair_growth, weight_gain, skin_darkening, fast_food, cycle_length
            )
            if risk_category == "High" and not st.session_state.get("already_redirected", False):
                st.session_state["switch_to_detailed"] = True
                st.session_state["already_redirected"] = True
                st.session_state.scroll_to_top_page3 = True  # Add this line to trigger scroll
                st.rerun()

    if st.session_state.get("symptom_form_submitted_page2", False) and st.session_state["recommendations_page2"]:
        st.markdown(st.session_state["recommendations_page2"], unsafe_allow_html=True)
        if st.session_state["risk_category_page2"] in ["Low", "Moderate"]:
            st.markdown('<div class="email-section">', unsafe_allow_html=True)
            handle_report_output(
                st.session_state["symptom_data"],
                st.session_state["risk_category_page2"],
                st.session_state["username"],
                st.session_state["user_id"],
                "page2"
            )
            st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("📜 Your Symptom History"):
        history = get_symptom_history(st.session_state["user_id"])
        if history:
            for entry in history:
                st.write(f"**Submitted at:** {entry['submitted_at']}")
                st.write(f"- Hair growth: {entry['hair_growth']}")
                st.write(f"- Weight gain: {entry['weight_gain']}")
                st.write(f"- Skin darkening: {entry['skin_darkening']}")
                st.write(f"- Fast food: {entry['fast_food']}")
                st.write(f"- Cycle length: {entry['cycle_length']} days")
                st.write("---")
        else:
            st.info("No previous submissions found.")

def page_detailed_analysis():
    # Initialize session state for scrolling
    if 'scroll_to_top_page3' not in st.session_state:
        st.session_state.scroll_to_top_page3 = False
    
    # Check if we need to scroll to top
    if st.session_state.scroll_to_top_page3:
        scroll_to_here(0, key='top_page3')
        st.session_state.scroll_to_top_page3 = False
    
    # CSS styles
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #ffffff 0%, #fff8fa 25%, #ffeef5 50%, #fff0f6 75%, #ffffff 100%);
        background-attachment: fixed;
        font-family: 'Quicksand', sans-serif, 'Helvetica Neue', sans-serif;
        color: #333;
        min-height: 100vh;
    }
    .custom-header {
        background: linear-gradient(90deg, #ff7b9c 0%, #ffb6c1 100%);
        color: #111;
        border-radius: 22px;
        box-shadow: 0 8px 24px rgba(255, 179, 198, 0.18);
        padding: 1.5rem 0;
        text-align: center;
        font-size: 2.5rem;
        font-weight: 600;
    }
    .section-card {
        background: #ffffff;
        border-radius: 18px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08), 0 2px 4px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(255, 105, 180, 0.15);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .phase-display {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(255, 248, 250, 0.98));
        backdrop-filter: blur(20px);
        border-radius: 25px;
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
        border: 2px solid rgba(255, 105, 180, 0.3);
        box-shadow: 0 15px 35px rgba(255, 105, 180, 0.2);
        position: relative;
        overflow: hidden;
    }
    .phase-title {
        font-size: 2.0rem !important;
        font-weight: 800;
        margin-bottom: 0.5rem;
        color: #000000 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .phase-description {
        font-size: 1.15rem;
        color: #000000 !important;
        line-height: 1.6;
        margin-bottom: 0.5rem;
    }
    .low-risk {
        background: linear-gradient(135deg, #d4edda, #c3e6cb) !important;
        border: 2px solid #28a745 !important;
        color: #155724 !important;
    }
    .moderate-risk {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7) !important;
        border: 2px solid #ffc107 !important;
        color: #856404 !important;
    }
    .high-risk {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb) !important;
        border: 2px solid #dc3545 !important;
        color: #721c24 !important;
    }
    .symptom-box {
        background: linear-gradient(135deg, #fff0f3, #ffe4e8);
        border-radius: 20px;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0 0.5rem 0;
        border: 1px solid rgba(255, 105, 180, 0.3);
        box-shadow: 0 8px 24px rgba(0,0,0,0.06);
    }
    .symptom-box ul { margin: 0.5rem 0 0 1.25rem; }
    .email-section {
        background: linear-gradient(135deg, #fff0f3, #ffe4e8);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 105, 180, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    .recommendation-title {
        color: #7e0000;
        font-size: 1.5rem;
        text-align: center;
        margin: 1rem 0 0.5rem 0;
    }
    .recommendation-disclaimer {
        color: #6c757d;
        font-style: italic;
        text-align: center;
        margin-top: 1rem;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Safe defaults
    st.session_state.setdefault("symptom_data", {})
    st.session_state.setdefault("risk_category_page3", None)
    st.session_state.setdefault("detailed_form_submitted", False)
    st.session_state.setdefault("detailed_input_data", {})
    st.session_state.setdefault("report_submitted_page3", False)
    st.session_state.setdefault("doctor_feedback", "No")
    st.session_state.setdefault("prediction_data", {})

    # ---------- header ----------
    st.markdown('<div class="custom-header">Detailed PCOS Analysis</div>', unsafe_allow_html=True)
    st.info("ℹ️ Hey there! Please do not worry, this is just a detailed analysis to help you understand your risk level better. Let's get started!")

    # Check if report form was submitted
    report_form_key = "email_form_page3"
    if f"{report_form_key}_submitted" in st.session_state:
        if st.session_state.get("prediction_data"):
            display_recommendations()
            st.markdown('<div class="email-section">', unsafe_allow_html=True)
            handle_report_output(
                st.session_state["detailed_input_data"],
                st.session_state["risk_category_page3"],
                st.session_state["username"],
                st.session_state["user_id"],
                "page3"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        return

    # ---------- Input UI ----------
    with st.container():
        st.markdown("<h3>Personal Information</h3>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Your age (years)", min_value=10, max_value=60, value=25, key="age_page3")
        with col2:
            weight = st.number_input("Your weight (kg)", min_value=30.0, max_value=200.0, value=60.0, key="weight_page3")
        with col3:
            height = st.number_input("Your height (cm)", min_value=100.0, max_value=220.0, value=160.0, key="height_page3")
        bmi = round(weight / ((height / 100) ** 2), 2)
        st.markdown(f"<p><strong>Your BMI:</strong> {bmi}</p>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<h3>Menstrual & Reproductive History</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            marriage_years = st.number_input("Years married (if applicable)", min_value=0, max_value=30, value=0, key="marriage_years_page3")
            pregnant = st.radio("Are you currently pregnant?", ["No", "Yes"], key="pregnant_page3")
        with col2:
            abortions = st.number_input("Number of previous pregnancy losses (if any)", min_value=0, max_value=10, value=0, key="abortions_page3")

    with st.container():
        st.markdown("<h3>Physical Measurements</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            hip = st.number_input("Hip circumference (inches)", min_value=20.0, max_value=60.0, value=36.0, key="hip_page3")
        with col2:
            waist = st.number_input("Waist circumference (inches)", min_value=20.0, max_value=60.0, value=30.0, key="waist_page3")
        wh_ratio = round(waist / hip, 2)
        st.markdown(f"<p><strong>Waist:Hip Ratio:</strong> {wh_ratio}</p>", unsafe_allow_html=True)

    # Symptoms recap
    symptoms = st.session_state.get("symptom_data", {})
    st.markdown(f"""
        <div class='symptom-box'>
            <p>Here's what you shared earlier:</p>
            <ul>
                <li>Hair growth: {symptoms.get('hair growth(Y/N)', 'N/A')}</li>
                <li>Weight gain: {symptoms.get('Weight gain(Y/N)', 'N/A')}</li>
                <li>Skin darkening: {symptoms.get('Skin darkening (Y/N)', 'N/A')}</li>
                <li>Fast food consumption: {symptoms.get('Fast food (Y/N)', 'N/A')}</li>
                <li>Cycle length (days): {symptoms.get('Cycle length(days)', 'N/A')}</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    # Lifestyle toggles
    col1, col2, col3 = st.columns(3)
    with col1:
        hair_loss = st.radio("Hair thinning/loss?", ["No", "Yes"], key="hair_loss_page3")
    with col2:
        pimples = st.radio("Acne/pimples?", ["No", "Yes"], key="pimples_page3")
    with col3:
        exercise = st.radio("Exercise regularly?", ["No", "Yes"], key="exercise_page3")

    # Optional clinical tests
    has_tests = st.radio("Do you have recent clinical test results to share?", ["No", "Yes"], key="has_tests_page3")
    clinical_inputs = {}
    
    # Doctor feedback question - ALWAYS show this
    doctor_feedback = st.radio("Have you discussed these results with a doctor?", ["No", "Yes"], 
                              key="doctor_feedback_page3")
    st.session_state.doctor_feedback = doctor_feedback
    
    if has_tests == "Yes":
        with st.container():
            st.markdown("<h3>🩺 Clinical Test Results</h3>", unsafe_allow_html=True)
            st.info("ℹ️ If you don't have values, leave defaults (dataset averages).")
            tab1, tab2, tab3 = st.tabs(["Blood Tests", "Vital Signs", "Ultrasound"])
            with tab1:
                clinical_inputs["Blood Group"] = st.selectbox("Your blood group", ["A", "B", "AB", "O"], key="blood_group_page3")
                clinical_inputs["Hb(g/dl)"] = st.number_input("Hemoglobin (g/dl)", min_value=5.0, max_value=20.0, value=13.0, key="hb_page3")
                clinical_inputs["FSH(mIU/mL)"] = st.number_input("FSH", min_value=0.0, max_value=20.0, value=6.0, key="fsh_page3")
                clinical_inputs["LH(mIU/mL)"] = st.number_input("LH", min_value=0.0, max_value=20.0, value=7.0, key="lh_page3")
                clinical_inputs["TSH (mIU/L)"] = st.number_input("TSH", min_value=0.0, max_value=10.0, value=2.5, key="tsh_page3")
                clinical_inputs["AMH(ng/mL)"] = st.number_input("AMH", min_value=0.0, max_value=10.0, value=3.0, key="amh_page3")
                clinical_inputs["PRL(ng/mL)"] = st.number_input("PRL", min_value=0.0, max_value=50.0, value=12.0, key="prl_page3")
                clinical_inputs["Vit D3 (ng/mL)"] = st.number_input("Vit D3", min_value=0.0, max_value=100.0, value=30.0, key="vitd3_page3")
                clinical_inputs["PRG(ng/mL)"] = st.number_input("Progesterone", min_value=0.0, max_value=30.0, value=10.0, key="prg_page3")
                clinical_inputs["RBS(mg/dl)"] = st.number_input("RBS", min_value=50, max_value=300, value=90, key="rbs_page3")
            with tab2:
                clinical_inputs["Pulse rate(bpm)"] = st.number_input("Pulse rate", min_value=50, max_value=120, value=75, key="pulse_page3")
                clinical_inputs["RR (breaths/min)"] = st.number_input("Resp rate", min_value=12, max_value=30, value=20, key="rr_page3")
                clinical_inputs["BP _Systolic (mmHg)"] = st.number_input("Systolic BP", min_value=80, max_value=200, value=120, key="sysbp_page3")
                clinical_inputs["BP _Diastolic (mmHg)"] = st.number_input("Diastolic BP", min_value=40, max_value=120, value=80, key="diabp_page3")
            with tab3:
                clinical_inputs["Follicle No. (L)"] = st.number_input("Follicles (L)", min_value=0, max_value=25, value=8, key="folL_page3")
                clinical_inputs["Follicle No. (R)"] = st.number_input("Follicles (R)", min_value=0, max_value=25, value=8, key="folR_page3")
                clinical_inputs["Avg. F size (L) (mm)"] = st.number_input("F size L", min_value=0.0, max_value=30.0, value=10.0, key="fsizeL_page3")
                clinical_inputs["Avg. F size (R) (mm)"] = st.number_input("F size R", min_value=0.0, max_value=30.0, value=10.0, key="fsizeR_page3")
                clinical_inputs["Endometrium (mm)"] = st.number_input("Endometrium", min_value=0.0, max_value=20.0, value=8.0, key="endo_page3")

    # Predict button
    if st.button("🔍 Predict My PCOS Risk", key="predict_risk_page3"):
        with st.spinner("Analyzing your information..."):
            def convert_yn_to_binary(v):
                return 1 if v == "Yes" else 0

            # build non-clinical inputs using Page 2 symptoms safely
            nc_inputs = {
                "Age (yrs)": age,
                "Weight (Kg)": weight,
                "Height(Cm)": height,
                "BMI": bmi,
                "Cycle length(days)": symptoms.get("Cycle length(days)", 28),
                "Marriage Status (Yrs)": marriage_years,
                "Hip(inch)": hip,
                "Waist(inch)": waist,
                "Waist:Hip Ratio": wh_ratio,
                "Pregnant(Y/N)": convert_yn_to_binary(pregnant),
                "No. of Abortions": abortions,
                "Weight gain(Y/N)": convert_yn_to_binary(symptoms.get("Weight gain(Y/N)", "No")),
                "hair growth(Y/N)": convert_yn_to_binary(symptoms.get("hair growth(Y/N)", "No")),
                "Skin darkening (Y/N)": convert_yn_to_binary(symptoms.get("Skin darkening (Y/N)", "No")),
                "Hair loss(Y/N)": convert_yn_to_binary(hair_loss),
                "Pimples(Y/N)": convert_yn_to_binary(pimples),
                "Fast food (Y/N)": convert_yn_to_binary(symptoms.get("Fast food (Y/N)", "No")),
                "Reg.Exercise(Y/N)": convert_yn_to_binary(exercise)
            }

            # merge clinical if provided
            if has_tests == "Yes":
                clinical_inputs["FSH/LH"] = round(
                    (clinical_inputs.get("FSH(mIU/mL)", 0) / clinical_inputs.get("LH(mIU/mL)", 1)
                    if clinical_inputs.get("LH(mIU/mL)", 1) != 0 else 0), 
                    2  # Round to 2 decimal places
                )
                clinical_inputs["Blood Group"] = {"A": 11, "B": 13, "AB": 15, "O": 14}.get(clinical_inputs.get("Blood Group"), 14)
                full_inputs = {**nc_inputs, **clinical_inputs}
            else:
                full_inputs = nc_inputs

            # load model via factory
            try:
                model_key = "all" if has_tests == "Yes" else "nc"
                model = ModelFactory.get_model(model_key)
                expected_features = overall_features if has_tests == "Yes" else non_clinical_features
            except Exception as e:
                st.error(f"❌ Error loading model '{model_key}': {e}")
                return

            if model is None:
                st.error("❌ Model could not be loaded or is not fitted.")
                return

            # dataframe in exact feature order
            input_data = {f: full_inputs.get(f, 0) for f in expected_features}
            input_df = pd.DataFrame([input_data], columns=expected_features)

            # prediction
            try:
                prob = model.predict_proba(input_df)[0][1] * 100
            except NotFittedError as nf:
                st.error("Prediction failed because the model is not fitted. Please ensure the saved file is a fitted classifier.")
                return
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                return

            # category thresholds (same logic)
            category = "Low Risk" if prob < 30 else "Moderate Risk" if prob < 70 else "High Risk"
            
            # After prediction, store the category and data
            st.session_state["risk_category_page3"] = category
            st.session_state["detailed_form_submitted"] = True
            st.session_state["detailed_input_data"] = input_data
            st.session_state["report_submitted_page3"] = False
            
            # Store only the data needed to regenerate recommendations, not HTML
            st.session_state["prediction_data"] = {
                "category": category,
                "doctor_feedback": doctor_feedback
            }
            
            # Display the recommendations immediately
            display_recommendations()
            
            # Show report options
            st.markdown('<div class="email-section">', unsafe_allow_html=True)
            handle_report_output(
                st.session_state["detailed_input_data"],
                st.session_state["risk_category_page3"],
                st.session_state["username"],
                st.session_state["user_id"],
                "page3"
            )
            st.markdown('</div>', unsafe_allow_html=True)

            # --- ADD HISTORY SECTION HERE TOO ---
            with st.expander("📜 Your Detailed Assessment History"):
                history = get_detailed_history(st.session_state["user_id"])
                if history:
                    for entry in history:
                        st.write(f"**Submitted at:** {entry['submitted_at']}")
                        st.write(f"**Risk Category:** {entry['risk_category']}")
                        st.write(f"**Age:** {entry['age']} | **BMI:** {entry['bmi']} | **Cycle Length:** {entry['cycle_length']} days")
                        st.write("---")
                else:
                    st.info("No previous detailed assessments found.")

    # Display recommendations if they exist from previous prediction
    elif st.session_state.get("detailed_form_submitted", False) and st.session_state.get("prediction_data"):
        display_recommendations()
        st.markdown('<div class="email-section">', unsafe_allow_html=True)
        handle_report_output(
            st.session_state["detailed_input_data"],
            st.session_state["risk_category_page3"],
            st.session_state["username"],
            st.session_state["user_id"],
            "page3"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # --- ADD HISTORY SECTION INSIDE HERE ---
        with st.expander("📜 Your Detailed Assessment History"):
            history = get_detailed_history(st.session_state["user_id"])
            if history:
                for entry in history:
                    st.write(f"**Submitted at:** {entry['submitted_at']}")
                    st.write(f"**Risk Category:** {entry['risk_category']}")
                    st.write(f"**Age:** {entry['age']} | **BMI:** {entry['bmi']} | **Cycle Length:** {entry['cycle_length']} days")
                    st.write("---")
            else:
                st.info("No previous detailed assessments found.")

# NEW FUNCTION: Display recommendations based on stored data
def display_recommendations():
    if not st.session_state.get("prediction_data"):
        return
        
    data = st.session_state["prediction_data"]
    category = data["category"]
    doctor_feedback = data["doctor_feedback"]
    
    # Determine risk class and icon color
    if "high" in category.lower():
        risk_class = "high-risk"
        icon_color = "#dc3545"
        risk_title = "🚨 High Risk 🚨"
    elif "moderate" in category.lower():
        risk_class = "moderate-risk"
        icon_color = "#ffc107"
        risk_title = "⚠️ Moderate Risk ⚠️"
    else:
        risk_class = "low-risk"
        icon_color = "#28a745"
        risk_title = "✅ Low Risk ✅"
    
    risk_desc = f"Your risk category is detected as {category.lower()}. Here are some recommendations."
    
    # Display the risk header
    st.markdown(f"""
    <div class="phase-display {risk_class}" style="margin: 2rem 0; padding: 1rem; border-radius: 10px; text-align: center; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-left: 5px solid {icon_color};">
        <h3 class="phase-title" style="color: {icon_color}; margin-bottom: 0.5rem;">{risk_title}</h3>
        <p class="phase-description" style="color: #495057; margin: 0;">{risk_desc}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display recommendations title
    st.markdown('<div class="recommendation-title"><strong>Your Personalized Recommendations</strong></div>', unsafe_allow_html=True)

    # Display the appropriate recommendations
    if "High" in category:
        if doctor_feedback == "Yes":
            st.markdown("""
            - **Great! Since you've already discussed your results with a doctor, continue following their guidance.**
            - **Maintain your treatment plan and lifestyle modifications as recommended by your healthcare provider.**
            - **Keep regular follow-up appointments to monitor your progress and adjust treatment as needed.**
            - **Continue tracking symptoms and share any changes with your doctor at your next visit.**
            """)
        else:
            st.markdown("""
            - **Please arrange an appointment with a gynecologist/endocrinologist for confirmatory evaluation (hormone panel, ultrasound, metabolic screening).**
            - **Discuss management options such as combined OCPs, anti-androgens, or insulin-sensitizing therapy as appropriate.**
            - **Begin lifestyle actions now: whole foods, reduced refined sugar, regular moderate exercise (~150 min/week), sleep hygiene, stress reduction.**
            - **Track symptoms and cycle; bring notes to your consultation.**
            """)
    elif "Moderate" in category:
        if doctor_feedback == "Yes":
            st.markdown("""
            - **Good! You've taken the important step of discussing your results with a doctor.**
            - **Continue following their recommendations and maintain any prescribed treatments or lifestyle changes.**
            - **Keep your doctor informed about any changes in symptoms or concerns you may have.**
            - **Schedule follow-up appointments as recommended to monitor your progress.**
            """)
        else:
            st.markdown("""
            - **Book a non-urgent review with your clinician to discuss symptoms and consider targeted tests.**
            - **Keep a simple symptom/cycle diary to monitor trends.**
            - **Maintain balanced nutrition, regular activity, and stress management; re-assess if symptoms change.**
            """)
    else:  # Low Risk
        if doctor_feedback == "Yes":
            st.markdown("""
            - **Excellent! You've confirmed your low risk status with a healthcare professional.**
            - **Continue with your healthy habits and maintain regular check-ups as recommended.**
            - **Remember that even with low risk, it's good to stay vigilant about your health.**
            """)
        else:
            st.markdown("""
            - **Continue to monitor your cycle and symptoms; re-check if anything changes.**
            - **Keep up healthy habits: whole foods, regular movement, sleep, and stress care.**
            - **Consult a clinician if new symptoms develop or persist.**
            """)
    
    # Disclaimer
    st.markdown('<div class="recommendation-disclaimer">💡 These tips support care but don\'t replace professional medical advice.</div>', unsafe_allow_html=True)
# ----------------- SESSION INITIALIZATION -----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "current_page" not in st.session_state:
    st.session_state.current_page = "Introduction"
if "nav_radio" not in st.session_state:
    st.session_state["nav_radio"] = "Introduction"

# ----------------- LOGIN CHECK -----------------
if not st.session_state.logged_in:
    page_login_register()
    st.stop()

username = st.session_state.username

# ----------------- DYNAMIC PAGE LIST -----------------
# Determine which pages should be available based on risk category
available_pages = ["Introduction", "Phase & Risk Detection"]
if st.session_state.get("risk_category_page2") in ["High"]:
    available_pages.append("Detailed Analysis")

# ----------------- HANDLE AUTO-REDIRECT LOGIC (Runs BEFORE sidebar) -----------------
if st.session_state.get("_force_nav_to_detailed", False) and "Detailed Analysis" in available_pages:
    # Clear the redirect trigger flag
    del st.session_state["_force_nav_to_detailed"]
    # Update session state to target Detailed Analysis
    st.session_state["nav_radio"] = "Detailed Analysis"
    # Prevent infinite loops or repeated redirects
    st.session_state["already_redirected"] = True
    # Force the app to reload and display the correct page
    st.rerun()

# Replace the sidebar section in your code with this
with st.sidebar:
    st.title("Navigation")
    st.markdown(
        f"<div style='font-size:0.95rem; color:#388e3c; background:rgba(76,175,80,0.08); "
        f"border-radius:8px; padding:0.6rem 1rem; margin-bottom:0.7rem;'>"
        f"🔓 Logged in as: <b>{username}</b>"
        "</div>",
        unsafe_allow_html=True
    )

    # Dropdown for navigation
    selected_page = st.selectbox(
        "Navigate",
        available_pages,
        key="nav_radio",
        format_func=lambda x: x,
        disabled=False
    )

    st.markdown("---")
    if st.button("Log Out", key="logout_btn"):
        st.session_state.clear()
        st.session_state.logged_in = False
        st.success("You have been logged out.")
        st.rerun()

# Use selected_page directly!
if selected_page == "Introduction":
    page_introduction()
elif selected_page == "Phase & Risk Detection":
    page_phase_risk()
elif selected_page == "Detailed Analysis":
    page_detailed_analysis()
else:
    st.warning(f"Unknown page selected: {selected_page}. Redirecting to Introduction.")
    st.session_state["nav_radio"] = "Introduction"
    st.rerun()
