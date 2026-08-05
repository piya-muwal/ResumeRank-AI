import streamlit as st
import pandas as pd
import os
import io
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import uuid
from datetime import datetime
import sqlite3
import re

# --- Streamlit Page Config ---
st.set_page_config(                                
    page_title="ResumeRank AI", 
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Database Setup ---
def init_db():
    """Initialize SQLite database with necessary tables"""
    conn = sqlite3.connect('Resume.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        email TEXT PRIMARY KEY,
        password TEXT NOT NULL,
        name TEXT,
        job_title TEXT,
        company TEXT,
        date_joined TEXT,
        last_login TEXT
    )
    ''')
    
    # Create ranking history table
    c.execute('''
    CREATE TABLE IF NOT EXISTS ranking_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        job_title TEXT,
        description TEXT,
        results TEXT,
        FOREIGN KEY (email) REFERENCES users (email)
    )
    ''')
    
    conn.commit()
    conn.close()

# --- Initialize Session State ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    st.session_state["user_email"] = None
    st.session_state["user_name"] = None
    st.session_state["profile_tab"] = "profile"
    st.session_state["current_page"] = "login"  # Default page: login, register, dashboard, profile

# --- Security Functions ---
def hash_password(password, salt=None):
    """Hash a password for storing."""
    if salt is None:
        salt = uuid.uuid4().hex
    hashed = hashlib.sha256(salt.encode() + password.encode()).hexdigest()
    return f"{salt}${hashed}"

def verify_password(stored_password, provided_password):
    """Verify a stored password against one provided by user"""
    salt, hashed = stored_password.split('$')
    return hashed == hashlib.sha256(salt.encode() + provided_password.encode()).hexdigest()

# --- User Management Functions ---
def save_user(email, password, name=""):
    """Registers a new user in the database."""
    conn = sqlite3.connect('Resume.db')
    c = conn.cursor()
    
    # Check if user exists
    c.execute("SELECT email FROM users WHERE email = ?", (email,))
    if c.fetchone():
        conn.close()
        return False  # User already exists
    
    # Hash the password
    hashed_password = hash_password(password)
    
    # Create new user with timestamp
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute(
        "INSERT INTO users VALUES (?, ?, ?, ?, ?, ?, ?)",
        (email, hashed_password, name, "", "", current_date, current_date)
    )
    
    conn.commit()
    conn.close()
    return True

def authenticate_user(email, password):
    """Authenticate a user with email and password."""
    conn = sqlite3.connect('Resume.db')
    c = conn.cursor()
    
    c.execute("SELECT password FROM users WHERE email = ?", (email,))
    result = c.fetchone()
    
    if not result:
        conn.close()
        return False
    
    stored_password = result[0]
    
    if verify_password(stored_password, password):
        # Update last login time
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("UPDATE users SET last_login = ? WHERE email = ?", (current_date, email))
        conn.commit()
        conn.close()
        return True
    
    conn.close()
    return False

def update_profile(email, name, job_title, company):
    """Update user profile information."""
    conn = sqlite3.connect('Resume.db')
    c = conn.cursor()
    
    c.execute(
        "UPDATE users SET name = ?, job_title = ?, company = ? WHERE email = ?",
        (name, job_title, company, email)
    )
    
    conn.commit()
    conn.close()
    return True

def get_user_profile(email):
    """Get user profile data."""
    conn = sqlite3.connect('Resume.db')
    c = conn.cursor()
    
    c.execute(
        "SELECT email, name, job_title, company, date_joined, last_login FROM users WHERE email = ?",
        (email,)
    )
    
    result = c.fetchone()
    conn.close()
    
    if not result:
        return None
    
    return {
        "email": result[0],
        "name": result[1],
        "job_title": result[2],
        "company": result[3],
        "date_joined": result[4],
        "last_login": result[5]
    }

def change_password(email, current_password, new_password):
    """Change user password."""
    conn = sqlite3.connect('Resume.db')
    c = conn.cursor()
    
    c.execute("SELECT password FROM users WHERE email = ?", (email,))
    result = c.fetchone()
    
    if not result:
        conn.close()
        return False, "User not found"
    
    stored_password = result[0]
    
    if not verify_password(stored_password, current_password):
        conn.close()
        return False, "Current password is incorrect"
    
    # Hash the new password
    hashed_password = hash_password(new_password)
    
    # Update password
    c.execute("UPDATE users SET password = ? WHERE email = ?", (hashed_password, email))
    conn.commit()
    conn.close()
    
    return True, "Password changed successfully"

# --- Resume History Functions ---
def save_ranking_history(email, job_title, description, results):
    """Save resume ranking history for the user."""
    conn = sqlite3.connect('Resume.db')
    c = conn.cursor()
    
    # Create new history entry
    c.execute(
        "INSERT INTO ranking_history (email, timestamp, job_title, description, results) VALUES (?, ?, ?, ?, ?)",
        (
            email,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            job_title,
            description,
            results.to_json()
        )
    )
    
    conn.commit()
    conn.close()

def get_user_history(email):
    """Get resume ranking history for the user."""
    conn = sqlite3.connect('Resume.db')
    
    # Get all history records for the user
    query = "SELECT id, timestamp, job_title, description, results FROM ranking_history WHERE email = ? ORDER BY timestamp DESC"
    history_df = pd.read_sql_query(query, conn, params=(email,))
    
    conn.close()
    
    return history_df

# --- Resume Processing Functions ---
def extract_text_from_pdf(file):
    """Extracts text from an uploaded PDF file."""
    try:
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip() if text else "No readable text found."
    except Exception as e:
        return f"Error extracting text: {str(e)}"

SKILL_ALIASES = {
    "python": ["python"],
    "java": ["java"],
    "javascript": ["javascript", "java script", "js", "es6"],
    "typescript": ["typescript", "type script"],
    "c++": ["c++", "cpp"],
    "c#": ["c#", "c sharp"],
    "sql": ["sql", "structured query language"],
    "html": ["html", "html5"],
    "css": ["css", "css3"],
    "react": ["react", "react.js", "reactjs"],
    "angular": ["angular", "angular.js", "angularjs"],
    "node.js": ["node.js", "nodejs", "node js"],
    "express.js": ["express", "express.js", "expressjs"],
    "django": ["django"],
    "flask": ["flask"],
    "fastapi": ["fastapi", "fast api"],
    "mongodb": ["mongodb", "mongo db"],
    "mysql": ["mysql", "my sql"],
    "postgresql": ["postgresql", "postgres", "postgre sql"],
    "sqlite": ["sqlite", "sqlite3"],
    "aws": ["aws", "amazon web services"],
    "azure": ["azure", "microsoft azure"],
    "gcp": ["gcp", "google cloud", "google cloud platform"],
    "docker": ["docker", "containerization", "containers"],
    "kubernetes": ["kubernetes", "k8s"],
    "git": ["git", "version control"],
    "github": ["github", "git hub"],
    "linux": ["linux", "ubuntu"],
    "excel": ["excel", "microsoft excel", "ms excel"],
    "power bi": ["power bi", "powerbi"],
    "tableau": ["tableau"],
    "pandas": ["pandas"],
    "numpy": ["numpy", "numpy"],
    "matplotlib": ["matplotlib"],
    "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "tensorflow": ["tensorflow", "tensor flow"],
    "pytorch": ["pytorch", "torch"],
    "keras": ["keras"],
    "machine learning": ["machine learning", "ml models", "ml model"],
    "deep learning": ["deep learning", "neural network", "neural networks"],
    "data science": ["data science", "data scientist"],
    "data analysis": ["data analysis", "data analytics", "analytics"],
    "artificial intelligence": ["artificial intelligence", "ai"],
    "nlp": ["nlp", "natural language processing", "text analytics"],
    "computer vision": ["computer vision", "image processing", "opencv"],
    "statistics": ["statistics", "statistical analysis"],
    "regression": ["regression", "linear regression", "logistic regression"],
    "classification": ["classification", "classifier"],
    "clustering": ["clustering", "k-means", "kmeans"],
    "streamlit": ["streamlit"],
    "rest api": ["rest api", "restful api", "restful services", "api development"],
    "salesforce": ["salesforce"],
    "apex": ["apex"],
    "lwc": ["lwc", "lightning web components"],
    "communication": ["communication", "presentation skills"],
    "leadership": ["leadership", "team lead"],
    "teamwork": ["teamwork", "team player", "collaboration"],
    "problem solving": ["problem solving", "analytical thinking", "debugging"]
}


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#.\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _contains_phrase(text, phrase):
    phrase = clean_text(phrase)
    if not phrase:
        return False
    pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
    return bool(re.search(pattern, text))


def extract_skills(text):
    cleaned = clean_text(text)
    found = []
    for canonical, aliases in SKILL_ALIASES.items():                       
        if any(_contains_phrase(cleaned, alias) for alias in aliases):
            found.append(canonical)
    return sorted(set(found))


def extract_keywords(job_description, limit=30):
    cleaned = clean_text(job_description)
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=limit,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9+#.]{1,}\b"
        )
        vectorizer.fit_transform([cleaned])
        return list(vectorizer.get_feature_names_out())
    except ValueError:
        return []


def calculate_ats_score(job_description, resume):
    """Return three clear measures: overall match, core skills, and resume quality."""
    cleaned_job = clean_text(job_description)
    cleaned_resume = clean_text(resume)

    try:
        tfidf = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True
        ).fit_transform([cleaned_job, cleaned_resume])
        semantic_similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    except ValueError:
        semantic_similarity = 0.0

    required_skills = extract_skills(job_description)
    resume_skills = extract_skills(resume)
    matched_skills = sorted(set(required_skills) & set(resume_skills))
    missing_skills = sorted(set(required_skills) - set(resume_skills))

    if required_skills:
        skill_score = len(matched_skills) / len(required_skills)
    else:
        # A general JD may not state named technologies. Avoid a misleading zero.
        skill_score = min(1.0, 0.55 * semantic_similarity + 0.45 * min(len(resume_skills) / 8, 1.0))

    job_keywords = extract_keywords(job_description)
    matched_keywords = [kw for kw in job_keywords if _contains_phrase(cleaned_resume, kw)]
    keyword_score = len(matched_keywords) / len(job_keywords) if job_keywords else semantic_similarity

    section_groups = {
        "Education": ["education", "academic qualification", "b.tech", "btech", "bachelor of technology"],
        "Experience": ["experience", "internship", "employment", "work history", "training"],
        "Projects": ["project", "projects", "project experience"],
        "Skills": ["skills", "technical skills", "technologies", "tools"]
    }
    sections_found = [
        name for name, aliases in section_groups.items()
        if any(alias in cleaned_resume for alias in aliases)
    ]
    section_score = len(sections_found) / len(section_groups)

    email_found = bool(re.search(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}", resume))
    phone_found = bool(re.search(r"(?:\+?91[\s-]?)?[6-9]\d{9}", re.sub(r"[\s()-]", "", resume)))
    profile_found = "linkedin" in cleaned_resume or "github" in cleaned_resume
    readable_length = 120 <= len(cleaned_resume.split()) <= 1800
    format_score = sum([email_found, phone_found, profile_found, readable_length]) / 4

    resume_quality = 0.60 * section_score + 0.40 * format_score
    role_relevance = 0.70 * semantic_similarity + 0.30 * keyword_score

    # Clear, non-duplicative overall score.
    overall = 0.50 * role_relevance + 0.35 * skill_score + 0.15 * resume_quality
    overall = round(max(0, min(overall * 100, 100)), 1)

    return {
        "ats_score": overall,
        "role_relevance": round(role_relevance * 100, 1),
        "skill_score": round(skill_score * 100, 1),
        "resume_quality": round(resume_quality * 100, 1),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "resume_skills": resume_skills,
        "matched_keywords": matched_keywords[:12],
        "sections_found": sections_found
    }


def get_hiring_recommendation(score):
    if score >= 75:
        return "🟢 Strong Match", "Shortlist for the next hiring stage."
    if score >= 60:
        return "🔵 Good Match", "Consider for interview after checking key requirements."
    if score >= 45:
        return "🟡 Partial Match", "Relevant profile, but important gaps should be reviewed."
    return "🔴 Low Match", "Not a priority for this job description."


def generate_resume_feedback(analysis):
    strengths, concerns, suggestions = [], [], []

    if analysis["skill_score"] >= 70:
        strengths.append("Good coverage of the core skills stated in the job description.")
    elif analysis["skill_score"] >= 40:
        strengths.append("Some relevant core skills are present.")
        concerns.append("Several stated job skills are not visible in the resume.")
    else:
        concerns.append("The resume shows limited evidence of the job's core skills.")

    if analysis["role_relevance"] >= 60:
        strengths.append("Projects and experience are reasonably aligned with the role.")
    else:
        concerns.append("Projects and experience are not strongly aligned with this role.")
        suggestions.append("Rewrite project and internship bullets around the responsibilities in this job description.")

    if analysis["resume_quality"] >= 70:
        strengths.append("The resume has a clear, ATS-readable structure.")
    else:
        concerns.append("The resume structure or contact details need improvement.")
        suggestions.append("Keep clear headings for Education, Skills, Projects and Experience, with email, phone and profile links.")

    if analysis["missing_skills"]:
        suggestions.append("Add evidence for applicable skills: " + ", ".join(analysis["missing_skills"][:8]) + ".")

    if not strengths:
        strengths.append("The resume was successfully parsed and evaluated.")
    if not concerns:
        concerns.append("No major concern detected from the submitted text.")
    if not suggestions:
        suggestions.append("Customise the summary and strongest project bullets for each application.")

    return strengths, concerns, suggestions


def rank_resumes(job_description, resumes):
    return [calculate_ats_score(job_description, resume) for resume in resumes]

# Add custom CSS for better styling
st.markdown("""
    <style>
        .stButton>button {
            background-color: #1e90ff;
            color: white;
            font-size: 16px;
            border-radius: 5px;
            padding: 10px;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #4682b4;
        }
        .sidebar .sidebar-content {
            padding: 20px;
        }
        .stTextInput>div>div>input {
            font-size: 16px;
            border-radius: 5px;
        }
        .stTextArea>div>div>textarea {
            font-size: 16px;
            border-radius: 5px;
        }
        .stTabs>div>div>button {
            font-size: 18px;
            font-weight: bold;
            color: #1e90ff;
        }
        .stTabs>div>div>button:hover {
            color: #4682b4;
        }
        .stExpander>div>div>button {
            font-size: 16px;
            font-weight: bold;
            color: #1e90ff;
        }
    </style>
""", unsafe_allow_html=True)

# --- Main Navigation --- 
def show_login_page():
    st.sidebar.title("📝 User Login")
    st.sidebar.markdown("### Please enter your credentials to login.")
    
    login_email = st.sidebar.text_input("📧 Email", key="login_email", placeholder="Enter your email")
    login_password = st.sidebar.text_input("🔑 Password", type="password", key="login_password", placeholder="Enter your password")
    
    st.sidebar.markdown("---")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🔐 Login", use_container_width=True):
            if authenticate_user(login_email, login_password):
                st.session_state["authenticated"] = True
                st.session_state["user_email"] = login_email
                profile = get_user_profile(login_email)
                st.session_state["user_name"] = profile["name"]
                st.session_state["current_page"] = "dashboard"
                st.rerun()
            else:
                st.sidebar.error("❌ Invalid email or password")
    
    with col2:
        if st.button("📝 Register", use_container_width=True):
            st.session_state["current_page"] = "register"
            st.rerun()

def show_register_page():
    st.sidebar.title("📝 User Registration")
    st.sidebar.markdown("### Create a new account to get started.")
    
    reg_email = st.sidebar.text_input("📧 Email*", key="reg_email", placeholder="Enter your email")
    reg_name = st.sidebar.text_input("👤 Full Name", key="reg_name", placeholder="Enter your full name")
    reg_password = st.sidebar.text_input("🔑 Password*", type="password", key="reg_password", placeholder="Enter your password")
    reg_confirm_password = st.sidebar.text_input("🔑 Confirm Password*", type="password", key="reg_confirm_password", placeholder="Confirm your password")
    
    st.sidebar.markdown("---")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("✅ Register", use_container_width=True):
            if not reg_email or not reg_password:
                st.sidebar.error("❌ Email and password are required")
            elif "@" not in reg_email or "." not in reg_email:
                st.sidebar.error("❌ Invalid email format")
            elif reg_password != reg_confirm_password:
                st.sidebar.error("❌ Passwords do not match")
            else:
                if save_user(reg_email, reg_password, reg_name):
                    st.sidebar.success("✅ Registration successful! You can now log in.")
                    st.session_state["current_page"] = "login"
                    st.rerun()
                else:
                    st.sidebar.warning("⚠ Email already registered. Please log in instead.")
                    st.session_state["current_page"] = "login"
                    st.rerun()
    
    with col2:
        if st.button("↩️ Back to Login", use_container_width=True):
            st.session_state["current_page"] = "login"
            st.rerun()

def show_profile_page():
    st.title("👤 User Profile")
    st.caption("👨‍💻 Developed by Piya Muwal")
    st.markdown("### Manage your profile information and preferences.")
    
    profile = get_user_profile(st.session_state["user_email"])
    if not profile:
        st.error("❌ Error loading profile data")
        return
    
    # Profile tabs
    profile_tab, password_tab, history_tab = st.tabs(["✏️ Edit Profile", "🔐 Change Password", "📊 History"])
    
    with profile_tab:
        st.subheader("Personal Information")
        
        name = st.text_input("Full Name", value=profile["name"] if profile["name"] else "")
        job_title = st.text_input("Job Title", value=profile["job_title"] if profile["job_title"] else "")
        company = st.text_input("Company", value=profile["company"] if profile["company"] else "")
        
        if st.button("💾 Save Profile"):
            if update_profile(profile["email"], name, job_title, company):
                st.session_state["user_name"] = name
                st.success("✅ Profile updated successfully!")
                st.rerun()
            else:
                st.error("❌ Error updating profile")
    
    with password_tab:
        st.subheader("Change Password")
        
        current_password = st.text_input("Current Password", type="password")
        new_password = st.text_input("New Password", type="password")
        confirm_new_password = st.text_input("Confirm New Password", type="password")
        
        if st.button("🔄 Update Password"):
            if not current_password or not new_password or not confirm_new_password:
                st.error("❌ All fields are required")
            elif new_password != confirm_new_password:
                st.error("❌ New passwords do not match")
            else:
                success, message = change_password(profile["email"], current_password, new_password)
                if success:
                    st.success(f"✅ {message}") 
                else:
                    st.error(f"❌ {message}")
    
    with history_tab:
        st.subheader("Resume Ranking History")
        
        history = get_user_history(profile["email"])
        if history.empty:
            st.info("📝 No ranking history found")
        else:
            for idx, row in history.iterrows():
                with st.expander(f"Job: {row['job_title']} - {row['timestamp']}"):
                    st.text_area(
                        "Job Description",
                        value=row["description"],
                        height=100,
                        disabled=True,
                        key=f"job_desc_{idx}"
                    )
                    try:
                        results = pd.read_json(io.StringIO(row["results"]))
                        st.dataframe(results, hide_index=True, use_container_width=True)
                    except (ValueError, TypeError):
                        st.warning("⚠ Error loading results data")



def show_dashboard():
    welcome_name = st.session_state["user_name"] or st.session_state["user_email"]

    # Title with gradient effect using HTML
    st.markdown("""
        <h2 style="
            background: -webkit-linear-gradient(45deg, #1FA2FF, #12D8FA);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            text-align: center;
            font-size: 2.5rem;">
            🚀 Welcome to ResumeRank AI
        </h2>
    """, unsafe_allow_html=True)

    st.markdown(f"<div style='text-align:center; font-size:18px;'>Welcome back, <b style='color:#4CAF50'>{welcome_name}</b> 👋</div>", unsafe_allow_html=True)
    st.markdown("### ")

    # --- Job Information Section ---
    with st.container():
        st.subheader("📄 Job Information")
        st.markdown("Fill in the job details to start screening candidates.")
        job_title = st.text_input("Job Title", placeholder="e.g., Trainee Engineer", label_visibility="visible")

    st.markdown("---")

    # --- Job Description & Resume Upload ---
    st.subheader("📋 Job Description & 📂 Resume Upload")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        job_description = st.text_area(
            "Job Description",
            placeholder="Paste or write the full job description here...",
            height=220,
            key="job_desc"
        )

    with col2:
        st.markdown("#### Upload Resumes")
        uploaded_files = st.file_uploader(
            "Select PDF resumes",
            type=["pdf"],
            accept_multiple_files=True,
            key="resume_files"
        )

        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} resume(s) uploaded successfully")

    st.markdown("---")

    # Optional: Next step / action button
    st.markdown("### Ready to rank candidates?")

    # --- Processing & Ranking ---
    if st.button("🔍 Rank Resumes", disabled=not (uploaded_files and job_description)):
        with st.spinner("🔍 Processing resumes..."):
            resumes = []
            file_names = []
            error_files = []
            
            # Process each resume
            for file in uploaded_files:
                text = extract_text_from_pdf(file)
                if "Error extracting text" in text:
                    error_files.append(file.name)
                else:
                    resumes.append(text)
                    file_names.append(file.name)
            
            if error_files:
                st.warning(f"⚠ Could not process {len(error_files)} files: {', '.join(error_files)}")
            
            if resumes:
                analyses = rank_resumes(
                    job_description,
                    resumes
                )

                ranked_resumes = sorted(
                    zip(
                        file_names,
                        resumes,
                        analyses
                    ),
                    key=lambda item: item[2]["ats_score"],
                    reverse=True
                )

                results_df = pd.DataFrame({
                    "Rank": range(
                        1,
                        len(ranked_resumes) + 1
                    ),

                    "Resume Name": [
                        name
                        for name, _, analysis
                        in ranked_resumes
                    ],

                    "ATS Score": [
                        f"{analysis['ats_score']}%"
                        for _, _, analysis
                        in ranked_resumes
                    ],

                    "Core Skills": [
                        f"{analysis['skill_score']}%"
                        for _, _, analysis
                        in ranked_resumes
                    ],

                    "Role Relevance": [
                        f"{analysis['role_relevance']}%"
                        for _, _, analysis
                        in ranked_resumes
                    ],

                    "Resume Quality": [
                        f"{analysis['resume_quality']}%"
                        for _, _, analysis
                        in ranked_resumes
                    ],

                    "Raw Score": [
                        analysis["ats_score"]
                        for _, _, analysis
                        in ranked_resumes
                    ]
                })

                st.subheader("🏆 ATS Ranked Resumes")

                st.caption(
                    "Overall Match combines role relevance, core skills and resume quality. "
                    "Use it as a screening aid, not as the only hiring decision."
                )

                st.dataframe(
                    results_df.drop(
                        columns=["Raw Score"]
                    ),
                    hide_index=True,
                    use_container_width=True
                )

                st.subheader(
                    "🧠 Detailed Candidate Analysis"
                )

                for rank, item in enumerate(
                    ranked_resumes,
                    start=1
                ):
                    name, resume_text, analysis = item

                    with st.expander(
                        f"#{rank} {name} — "
                        f"ATS Score: "
                        f"{analysis['ats_score']}%"
                    ):
                        col1, col2, col3 = st.columns(3)

                        col1.metric(
                            "Overall Match",
                            f"{analysis['ats_score']}%"
                        )

                        col2.metric(
                            "Core Skills",
                            f"{analysis['skill_score']}%"
                        )

                        col3.metric(
                            "Resume Quality",
                            f"{analysis['resume_quality']}%"
                        )

                        left, right = st.columns(2)

                        with left:
                            st.markdown(
                                "### ✅ Skills Found"
                            )

                            if analysis["matched_skills"]:
                                st.success(
                                    ", ".join(
                                        analysis[
                                            "matched_skills"
                                        ]
                                    )
                                )
                            else:
                                st.info(
                                    "No required skills found."
                                )

                        with right:
                            st.markdown(
                                "### ❌ Missing Skills"
                            )

                            if analysis["missing_skills"]:
                                st.warning(
                                    ", ".join(
                                        analysis[
                                            "missing_skills"
                                        ]
                                    )
                                )
                            else:
                                st.success(
                                    "No required skills missing."
                                )

                        recommendation, recommendation_text = get_hiring_recommendation(
                            analysis["ats_score"]
                        )
                        strengths, concerns, suggestions = generate_resume_feedback(analysis)

                        st.markdown("### 🎯 Hiring Recommendation")
                        st.info(f"{recommendation} — {recommendation_text}")

                        feedback_left, feedback_right = st.columns(2)
                        with feedback_left:
                            st.markdown("### 💪 Strengths")
                            for point in strengths:
                                st.write(f"• {point}")

                            st.markdown("### ⚠️ Points to Review")
                            for point in concerns:
                                st.write(f"• {point}")

                        with feedback_right:
                            st.markdown("### 🛠️ Improvement Suggestions")
                            for point in suggestions:
                                st.write(f"• {point}")

                st.subheader(
                    "📊 Top Candidates Visualization"
                )

                top_n = min(
                    len(results_df),
                    10
                )

                chart_data = results_df.head(
                    top_n
                ).copy()

                st.bar_chart(
                    chart_data.set_index(
                        "Resume Name"
                    )["Raw Score"]
                )
                
                # Save ranking history
                save_ranking_history(
                    st.session_state["user_email"],
                    job_title if job_title else "Unnamed Job",
                    job_description,
                    results_df
                )
                
                # Download options
                col1, col2 = st.columns(2)
                with col1:
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download CSV", csv, "ranked_resumes.csv", "text/csv")
                with col2:
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        results_df.to_excel(writer, index=False)
                    buffer.seek(0)
                    st.download_button("📥 Download Excel", buffer, "ranked_resumes.xlsx", 
                                      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.error("❌ No valid resumes to process")

# --- App Sidebar ---
def render_sidebar():
    st.sidebar.markdown("""
<h2 style="
    text-align: center;
    font-weight: bold;
    font-size: 48px;
    background: linear-gradient(90deg, #4CAF50, #2196F3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
">
    ResumeRank AI
</h2>
                        """, unsafe_allow_html=True)
    
    if st.session_state["authenticated"]:
        st.sidebar.subheader(f"👤 {st.session_state['user_email']}")
        
        # Navigation
        st.sidebar.markdown("---")
        st.sidebar.subheader("📱 Navigation")
        
        if st.sidebar.button("🏠 Dashboard", use_container_width=True):
            st.session_state["current_page"] = "dashboard"
            st.rerun()
            
        if st.sidebar.button("👤 My Profile", use_container_width=True):
            st.session_state["current_page"] = "profile"
            st.rerun()
            
        # Logout Button
        st.sidebar.markdown("---")
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            st.session_state["authenticated"] = False
            st.session_state["user_email"] = None
            st.session_state["user_name"] = None
            st.session_state["current_page"] = "login"
            st.sidebar.success("👋 Logged out successfully!")
            st.rerun()

# --- Global Footer (outside sidebar) ---
def render_footer():
    st.markdown("""
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f1f1f1;
            color: #555;
            text-align: center;
            padding: 10px 0;
            font-size: 14px;
            border-top: 1px solid #ccc;
        }
        </style>
        <div class="footer">
            © 2026 ResumeRank AI • Developed by Piya Muwal
        </div>
    """, unsafe_allow_html=True)


# --- Main App Logic ---
def main():
    # Initialize database
    init_db()
    
    render_sidebar()
    
    if not st.session_state.get("authenticated", False):
        # Landing page for unauthenticated users
        st.markdown("""
<h1 style='
    text-align: left;
    font-weight: bold;
    font-size: 48px;
    background: linear-gradient(90deg, #4CAF50, #2196F3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
'>
📄 Welcome to ResumeRank AI
</h1>
""", unsafe_allow_html=True)
        st.subheader("Your AI-powered hiring assistant")

        st.markdown("""
        ### 🚀 Why Use ResumeRank AI?
        - 🔍 **Intelligent Resume Matching**: Find candidates who truly match your job criteria.
        - ⚡ **Boost Efficiency**: Save hours of manual screening.
        - 📈 **Data-Driven Ranking**: Make fair, unbiased decisions.
        - 🧾 **Track & Compare**: Store ranking history for better long-term hiring strategy.
        """)

        # Advanced section
        st.markdown("### 🛠️ Advanced Features")
        st.markdown("""
        - 🧠 **AI-Powered Resume Parsing**
        - 📊 **Similarity Score Visualizations**
        - 💾 **Exportable Reports**
        - 🗂️ **Job Description Templates**
        - 🔐 **Secure User Profiles**
        """)

        st.markdown("---")

        # Layout for login/register
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.session_state["current_page"] == "login":
                show_login_page()
        with col2:
            if st.session_state["current_page"] == "register":
                show_register_page()
    
    else:
        # Authenticated pages
        if st.session_state["current_page"] == "dashboard":
            show_dashboard()
        elif st.session_state["current_page"] == "profile":
            show_profile_page()

if __name__ == "__main__":
    main()