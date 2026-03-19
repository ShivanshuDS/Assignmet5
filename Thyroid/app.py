"""
ThyroidDetect — AI Thyroid Disease Predictor
=============================================
Author : Anmol Panchal | B.Tech CSE 2nd Year | Data Science
Email  : panchalji9705@gmail.com
Phone  : +91 77422 03252

HOW TO RUN:
  1. pip install flask scikit-learn pandas numpy
  2. Place thyroid_model.pkl in same folder as app.py
  3. python app.py
  4. Open http://localhost:5000
"""

import os
import pickle
import pandas as pd
from datetime import date
from flask import Flask, render_template, request, redirect, url_for, session, jsonify

# ══════════════════════════════════════════════════════════════════════════════
# APP SETUP
# ══════════════════════════════════════════════════════════════════════════════
app = Flask(__name__)
app.secret_key = 'thyroiddetect_anmol_2026'
BASE = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# Reads feature names DIRECTLY from model.feature_names_in_
# → no hardcoded column names → no mismatch errors ever again
# ══════════════════════════════════════════════════════════════════════════════
model        = None
MODEL_LOADED = False
MODEL_ERROR  = ''
FEATURES     = []

try:
    pkl_path = os.path.join(BASE, 'thyroid_model.pkl')
    with open(pkl_path, 'rb') as f:
        model = pickle.load(f)

    if hasattr(model, 'feature_names_in_'):
        FEATURES = list(model.feature_names_in_)
        print(f"[OK] Model loaded — {len(FEATURES)} features auto-detected:")
        for i, name in enumerate(FEATURES, 1):
            print(f"     {i:2}. {name}")
    else:
        # sklearn < 1.0 fallback — using notebook column order
        FEATURES = [
            'age', 'sex', 'on thyroxine', 'sick', 'pregnant',
            'thyroid surgery', 'I131 treatment', 'on antithyroid medication',
            'goitre', 'tumor', 'TSH', 'T3', 'TT4', 'T4U', 'FTI'
        ]
        print("[WARN] model.feature_names_in_ not found — using notebook fallback")

    MODEL_LOADED = True

except FileNotFoundError:
    MODEL_ERROR = "thyroid_model.pkl not found. Place it next to app.py and restart."
    print(f"[WARNING] {MODEL_ERROR}")
except Exception as ex:
    MODEL_ERROR = str(ex)
    print(f"[WARNING] Could not load model: {MODEL_ERROR}")

# ══════════════════════════════════════════════════════════════════════════════
# FORM → DATAFRAME CONVERTER
# HTML form fields use underscores (e.g. on_thyroxine)
# Dataset columns use spaces  (e.g. on thyroxine)
# This function maps form values → exact model column names automatically
# ══════════════════════════════════════════════════════════════════════════════

def _norm(s):
    """Normalise a string for loose matching: lowercase, underscores."""
    return s.strip().lower().replace(' ', '_').replace('-', '_')

# Build lookup: normalised_name → actual model column name
FEAT_MAP = {_norm(f): f for f in FEATURES}

def form_to_df(form):
    row = {}
    for norm_key, col_name in FEAT_MAP.items():
        raw = form.get(norm_key, '').strip()
        if raw == '':
            row[col_name] = 0  # default to 0 instead of raising
            continue
        try:
            row[col_name] = int(float(raw))
        except ValueError:
            if raw.upper() in ('M', 'MALE'):
                row[col_name] = 1
            elif raw.upper() in ('F', 'FEMALE'):
                row[col_name] = 0
            else:
                raise ValueError(f"Unrecognised value '{raw}' for '{col_name}'")

    return pd.DataFrame([row], columns=FEATURES)
# ══════════════════════════════════════════════════════════════════════════════
# IN-MEMORY HISTORY
# ══════════════════════════════════════════════════════════════════════════════
history_store = {}

def get_history():
    return history_store.get(session.get('username', ''), [])

def get_stats():
    h = get_history()
    return {
        'total':    len(h),
        'positive': sum(1 for r in h if r['result'] == 'P'),
        'negative': sum(1 for r in h if r['result'] == 'N'),
    }

def is_logged_in():
    return 'username' in session

# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return redirect(url_for('dashboard') if is_logged_in() else url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if is_logged_in():
        return redirect(url_for('dashboard'))
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if not username or not password:
            error = 'Please enter both username and password.'
        else:
            session['username'] = username
            return redirect(url_for('dashboard'))
    return render_template('login.html', error=error)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if not is_logged_in():
        return redirect(url_for('login'))

    result    = None
    error     = None
    form_data = {}

    if request.method == 'POST':
        form_data = request.form.to_dict()
        try:
            if MODEL_LOADED:
                # Build DataFrame — columns auto-matched to model
                input_df   = form_to_df(request.form)
                prediction = str(model.predict(input_df)[0])

                if hasattr(model, 'predict_proba'):
                    proba      = model.predict_proba(input_df)[0]
                    confidence = round(float(max(proba)) * 100, 1)
                else:
                    confidence = 88.0
            else:
                # Demo mode — simple TSH rule
                tsh_val    = float(request.form.get('tsh', 2.0))
                prediction = 'P' if (tsh_val < 0.4 or tsh_val > 4.0) else 'N'
                confidence = 78.5

            is_pos       = (prediction == 'P')
            result_label = 'Positive — Thyroid Disease Detected' if is_pos else 'Negative — No Thyroid Disease'
            result_class = 'positive' if is_pos else 'negative'

            result = {
                'raw':        prediction,
                'label':      result_label,
                'class':      result_class,
                'confidence': confidence,
                'demo':       not MODEL_LOADED,
            }

            # Save to history
            u = session['username']
            history_store.setdefault(u, [])
            history_store[u].append({
                'id':         len(history_store[u]),
                'age':        form_data.get('age', '—'),
                'sex':        'Male' if form_data.get('sex') == 'M' else 'Female',
                'tsh':        form_data.get('tsh', '—'),
                'result':     prediction,
                'label':      result_label,
                'class':      result_class,
                'confidence': confidence,
                'date':       date.today().strftime('%d %b %Y'),
            })

        except Exception as ex:
            error = f'Error: {str(ex)}'

    recent = list(reversed(get_history()[-6:]))
    return render_template('dashboard.html',
        result       = result,
        error        = error,
        form_data    = form_data,
        stats        = get_stats(),
        recent       = recent,
        username     = session['username'],
        model_loaded = MODEL_LOADED,
        model_error  = MODEL_ERROR,
        features     = FEATURES,
    )


@app.route('/history')
def history():
    if not is_logged_in():
        return redirect(url_for('login'))
    records = list(enumerate(get_history()))
    return render_template('history.html',
        records  = records,
        stats    = get_stats(),
        username = session['username'],
    )


@app.route('/delete_record/<int:index>', methods=['POST'])
def delete_record(index):
    if not is_logged_in():
        return redirect(url_for('login'))
    h = history_store.get(session['username'], [])
    if 0 <= index < len(h):
        h.pop(index)
    return redirect(url_for('history'))


@app.route('/clear_history', methods=['POST'])
def clear_history():
    if not is_logged_in():
        return redirect(url_for('login'))
    history_store[session['username']] = []
    return redirect(url_for('history'))


@app.route('/about')
def about():
    return render_template('about.html',
        username     = session.get('username'),
        model_loaded = MODEL_LOADED,
        features     = FEATURES,
    )


# ── DEBUG ENDPOINT — visit /debug/features to see what the model expects ───
@app.route('/debug/features')
def debug_features():
    return jsonify({
        'model_loaded':  MODEL_LOADED,
        'feature_count': len(FEATURES),
        'features':      FEATURES,
        'model_error':   MODEL_ERROR,
    })


if __name__ == '__main__':
    print("\n" + "="*55)
    print("  ThyroidDetect  →  http://localhost:5000")
    print("="*55 + "\n")
    app.run(debug=True, port=5000)