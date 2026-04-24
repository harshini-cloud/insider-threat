from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
import hashlib
import os
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
app.secret_key = os.urandom(24)

DB_NAME = "database.db"
MODEL_PATH = "mlp_model.pkl"

numeric_features = ['size', 'attachments', 'num_recipients', 'hour', 'day_of_week']
X_numeric_columns = None
scaler = None
tfidf = None
loaded_mlp = None

def load_email_models():
    global loaded_mlp, scaler, tfidf, X_numeric_columns
    try:
        with open(MODEL_PATH, "rb") as f:
            loaded_mlp = pickle.load(f)
        print("MLP model loaded successfully.")
        import joblib
        scaler = joblib.load('scaler.pkl')
        tfidf = joblib.load('tfidf.pkl')
        X_numeric_columns = joblib.load('X_numeric_columns.pkl')
        print("Scaler, TF-IDF, and X_numeric_columns loaded successfully.")
    except Exception as e:
        print(f"Failed to load models/scalers: {e}")
        loaded_mlp = None
        scaler = None
        tfidf = None
        X_numeric_columns = None

def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER NOT NULL,
                occupation TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                role TEXT DEFAULT 'user'
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                risk_score REAL,
                threat_level TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        ''')
        admin_pass = hashlib.sha256("admin".encode()).hexdigest()
        try:
            cursor.execute("INSERT INTO users (name, age, occupation, email, password, role) VALUES (?, ?, ?, ?, ?, ?)",
                           ("System Admin", 30, "Security Chief", "admin", admin_pass, "admin"))
            conn.commit()
        except sqlite3.IntegrityError:
            pass

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def build_email_features(form_data):
    global scaler, tfidf, X_numeric_columns
    if scaler is None or tfidf is None or X_numeric_columns is None:
        raise ValueError("Scaler, TF-IDF, or X_numeric_columns not loaded. Run load_email_models() first.")

    size_val = float(form_data['size'])
    attachments_val = float(form_data['attachments'])
    num_recipients_val = float(form_data['num_recipients'])
    hour_val = float(form_data['hour'])
    day_of_week_val = float(form_data['day_of_week'])
    content_val = form_data.get('content', '').strip()

    # Use exact 5 columns as fitted scaler expects
    user_data_numeric_values = [[
        size_val,
        attachments_val,
        num_recipients_val,
        hour_val,
        day_of_week_val
    ]]
    user_data_numeric = pd.DataFrame(user_data_numeric_values, columns=X_numeric_columns)

    # Scale using global pre-fit scaler
    user_data_scaled_array = scaler.transform(user_data_numeric)
    user_data_scaled_df = pd.DataFrame(user_data_scaled_array, columns=X_numeric_columns)

    # TF-IDF content
    if content_val:
        content_tfidf_user = tfidf.transform([content_val]).toarray()
        content_tfidf_df_user = pd.DataFrame(content_tfidf_user, columns=tfidf.get_feature_names_out())
    else:
        content_tfidf_df_user = pd.DataFrame(0.0, index=[0], columns=tfidf.get_feature_names_out())

    # Combine
    user_data_combined = pd.concat([
        user_data_scaled_df.reset_index(drop=True),
        content_tfidf_df_user.reset_index(drop=True)
    ], axis=1)

    return user_data_combined

def generate_explanations(form_data):
    if loaded_mlp is None:
        raise FileNotFoundError("MLP model not loaded.")

    X = build_email_features(form_data)

    # Make prediction
    prediction = loaded_mlp.predict(X)

    # Display result
    threat_level = "ANOMALY DETECTED" if prediction[0] == 1 else "NORMAL EMAIL"
    
    proba = loaded_mlp.predict_proba(X)[0]
    risk_score = float(prediction[0]) * 100

    return {
        'prediction': int(prediction[0]),
        'risk_score': risk_score,
        'threat_level': threat_level,
        'proba_normal': proba[0],
        'proba_anomaly': proba[1],
        'features': dict(zip(numeric_features, [form_data.get(f, '0') for f in numeric_features])),
        'content': form_data.get('content', '')
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        age = request.form['age']
        occupation = request.form['occupation']
        email = request.form['email']
        password = hash_password(request.form['password'])

        try:
            conn = get_db_connection()
            conn.execute("INSERT INTO users (name, age, occupation, email, password) VALUES (?, ?, ?, ?, ?)",
                         (name, age, occupation, email, password))
            conn.commit()
            conn.close()
            flash("Registration Successful! Please Login.", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Email already registered.", "error")
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = hash_password(request.form['password'])

        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE email = ? AND password = ?", (email, password)).fetchone()
        conn.close()

        if user:
            session['user_id'] = user['id']
            session['role'] = user['role']
            session['name'] = user['name']
            if user['role'] == 'admin':
                return redirect(url_for('admin_dashboard'))
            return redirect(url_for('home'))
        else:
            flash("Invalid Credentials", "error")

    return render_template('login.html')

@app.route('/home')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (session['user_id'],)).fetchone()
    conn.close()
    return render_template('home.html', user=user)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            results = generate_explanations(request.form)
            
            # Save to DB
            conn = get_db_connection()
            conn.execute("INSERT INTO predictions (user_id, risk_score, threat_level) VALUES (?, ?, ?)",
                         (session['user_id'], results['risk_score'], results['threat_level']))
            conn.commit()
            conn.close()
            
            return render_template('predict.html', results=results, form_data={}, model_loaded=True)
        except Exception as exc:
            flash(f"Prediction failed: {str(exc)}", "error")
            return render_template('predict.html', form_data=request.form, model_loaded=loaded_mlp is not None)

    return render_template('predict.html', form_data={}, model_loaded=loaded_mlp is not None)

@app.route('/admin')
def admin_dashboard():
    if 'user_id' not in session or session.get('role') != 'admin':
        flash("Access Denied", "error")
        return redirect(url_for('login'))

    conn = get_db_connection()
    users = conn.execute("SELECT * FROM users").fetchall()
    predictions = conn.execute("SELECT p.*, u.name as user_name FROM predictions p JOIN users u ON p.user_id = u.id ORDER BY p.timestamp DESC").fetchall()
    conn.close()
    return render_template('admin.html', users=users, predictions=predictions)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    init_db()
    load_email_models()
    app.run(debug=True)

