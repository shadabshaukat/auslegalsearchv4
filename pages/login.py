"""
Login and Signup page for AUSLegalSearchv2.
- Local email/password auth (register/login)
- Google Login placeholder (extend with OAuth as needed)
- All users stored in Postgres DB
"""

import streamlit as st
from db.store import (
    create_user, get_user_by_email, check_password, set_last_login,
    get_user_by_googleid
)
from sqlalchemy.exc import IntegrityError

# Remove sidebar on login page by hiding it with CSS
st.set_page_config(page_title="Login", layout="centered")
st.markdown("""
    <style>
    [data-testid="stSidebar"], .css-1lcbmhc, .css-164nlkn {display: none !important;}
    </style>
    """, unsafe_allow_html=True)

if "auth_mode" not in st.session_state:
    st.session_state["auth_mode"] = "login"  # or "signup"

def do_signup():
    st.markdown("#### Register New Account")
    email = st.text_input("Email Address", key="signup_email")
    password = st.text_input("Password", type="password", key="signup_pw")
    name = st.text_input("Your Name (optional)", key="signup_name")
    sub = st.button("Create Account")
    msg = ""
    if sub:
        if not (email and password):
            msg = "Email and password required."
        elif get_user_by_email(email):
            msg = "Email already registered."
        else:
            try:
                user = create_user(email=email, password=password, name=name or None)
                st.success("Account created. You can login now.")
                st.session_state["auth_mode"] = "login"
            except IntegrityError:
                msg = "Email already exists."
            except Exception as e:
                msg = f"Error: {e}"
    if msg:
        st.error(msg)
    st.markdown("Already have an account? [Login here](#)", unsafe_allow_html=True)
    if st.button("Switch to Login"):
        st.session_state["auth_mode"] = "login"
        st.experimental_rerun() if hasattr(st, 'experimental_rerun') else st.rerun()

def do_login():
    st.markdown("#### Login to your Account")
    email = st.text_input("Email Address", key="login_email")
    password = st.text_input("Password", type="password", key="login_pw")
    msg = ""
    login = st.button("Login")
    if login:
        user = get_user_by_email(email)
        if not user:
            msg = "No such user (check your email or sign up)."
        elif not user.password_hash:
            msg = "This user is registered via Google. Use Google login."
        elif not check_password(password, user.password_hash):
            msg = "Incorrect password."
        else:
            set_last_login(user.id)
            st.session_state["user"] = {
                "id": user.id, "email": user.email, "name": user.name,
                "registered_google": user.registered_google
            }
            st.success(f"Welcome, {user.email}!")
            # Redirect to chat page after successful login
            if hasattr(st, "switch_page"):
                st.switch_page("pages/chat.py")
            else:
                st.markdown("Login successful. Go to [Chat](/chat)")
                st.experimental_rerun() if hasattr(st, 'experimental_rerun') else st.rerun()
    if msg:
        st.error(msg)
    st.markdown("Don't have an account? [Sign up here](#)", unsafe_allow_html=True)
    if st.button("Switch to Signup"):
        st.session_state["auth_mode"] = "signup"
        st.experimental_rerun() if hasattr(st, 'experimental_rerun') else st.rerun()

st.markdown("### AUSLegalSearch Login")
st.markdown(
    """
    <small>
    You may use Email/Password or Google Login.<br>
    All credentials securely stored in PostgreSQL.<br>
    </small>
    """, unsafe_allow_html=True
)

if st.session_state["auth_mode"] == "signup":
    do_signup()
else:
    do_login()

st.divider()
st.markdown("#### Or sign in with Google")
st.info("Google Login integration requires project/client id setup. To enable, add your OAuth Client ID and use a library such as `streamlit-authenticator` or `streamlit-oauth`. Fill in Google credentials then link users via email in the `users` table.")
# Placeholder for Google login button
google_btn = st.button("Sign in with Google (not yet linked)")
if google_btn:
    st.warning("Google login not yet implemented. Admin must configure OAuth client/secret for this app. Email support if you need this enabled.")
