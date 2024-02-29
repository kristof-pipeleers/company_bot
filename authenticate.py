import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore, auth, storage
from company_search_bot import run_app

# Function to load CSS from a file and inject it into the app
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css('styles.css')

# Initialize Firebase
try:
    firebase_app = firebase_admin.get_app()
except ValueError:
    # Assuming 'st.secrets["SERVICE_ACCOUNT"]' is a valid Firebase service account config
    cred = credentials.Certificate(st.secrets["SERVICE_ACCOUNT"])
    firebase_app = firebase_admin.initialize_app(cred, {'storageBucket': 'socs-415712.appspot.com'})

# Firestore Database
db = firestore.client()

# Firebase Storage
bucket = storage.bucket()

# Constants
MIN_PASSWORD_LENGTH = 6

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# Page switch function
def switch_to_app_page(username=None):
    st.session_state['logged_in'] = True
    st.session_state['username'] = username
    st.rerun()

def switch_to_login_page():
    st.session_state['logged_in'] = False
    st.rerun()

def load_welcome():
    # Welcome Page Content
    st.title("🍃 advAI:green[CE]")
    st.markdown("""
        ### Wat kan ik voor je doen?
        
        - :office: **Zoek naar bedrijven:** Vind bedrijven in verschillende sectoren en locaties.
        - 📍 **Geomapping:** Ontvang een overzicht van de locaties waar de bedrijven die u zoekt zich bevinden.
        - 🦋 **Waardeketenmapping:** :red[Coming soon...]
        
        ### Hoe te beginnen?
        
        1. **Maak een account aan** of **log in** via de zijbalk.
        2. Volg de eenvoudige instructies om je zoekopdrachten te starten.
        3. Ontvang resultaten in enkele seconden.
        
        ### Waarom deze chatbot gebruiken?
        
        - 💡 **Snelle en nauwkeurige informatie:** Bespaar tijd met snelle en relevante zoekresultaten.
        - 📝 **Eenvoudige interactie:** Gebruiksvriendelijke interface met stapsgewijze begeleiding.
        - 📡 **externe data bronnen:** De chatbot maakt gebruik van meerdere databronnen en combineert de resultaten tot het optimale antwoord.
        
        Begin nu om de kracht van 🍃 :orange[advAI]:green[CE] te ervaren! 🚀
    """, unsafe_allow_html=True)
    
    image_urls = [
        'images/werecircle-logo.png',
        'images/greenaumatic-logo.png'
    ]

    cols = st.columns(len(image_urls))
    for col, url in zip(cols, image_urls):
        col.image(url, width=300)

    # Spacer
    st.markdown("---")

    # Additional instructions or information
    st.markdown("""
        Heb je hulp nodig of wil je meer weten? Neem gerust contact met ons op via [info@werecircle.be](mailto:info@werecircle.be).
        
        Volg ons ook op [sociale media](https://www.linkedin.com/company/werecircle/) voor de laatste updates en nuttige tips! 🌟
    """, unsafe_allow_html=True)

# Authentication Block
def show_auth():
    st.sidebar.title("🍃 advAI:green[CE]")
    choice = st.sidebar.radio('Login / Registreer', ['Login', 'Sign up'])

    email = st.sidebar.text_input('E-mailadres' , placeholder="example@random.com")
    password = st.sidebar.text_input('Wachtwoord', type='password', help=f'Het wachtwoord moet minstens {MIN_PASSWORD_LENGTH} tekens lang zijn.', placeholder='Voer uw wachtwoord in')

    if choice == 'Sign up':
        username = st.sidebar.text_input('Voer de gebruikersnaam van je app in', placeholder="Jan Jansen")
        submit = st.sidebar.button('Mijn account aanmaken')
        if submit:
            if len(password) < MIN_PASSWORD_LENGTH:
                st.sidebar.error(f'Het wachtwoord moet minstens {MIN_PASSWORD_LENGTH} tekens lang zijn.', icon='⚠️')
            else:
                if submit:
                    user = auth.create_user(email=email, password=password)
                    st.sidebar.success(' Je account is succesvol aangemaakt!', icon='✅')
                    st.balloons()
                    # Add user data to Firestore
                    user_data = {'username': username, 'email': email}
                    db.collection('users').document(user.uid).set(user_data)
                    st.title(f'👋 Welkom, {username}!')
                    st.info(' Login via de login checkbox', icon='👈')

    elif choice == 'Login':
        if st.sidebar.button('Login'):
            try:
                user = auth.get_user_by_email(email)
                # Assuming you have a 'users' collection with documents named by user UID
                user_data = db.collection('users').document(user.uid).get()
                print(user_data)
                if user_data.exists:
                    st.sidebar.success(f'Succesvol ingelogd', icon='✅')
                    username = user_data.get('username')
                    switch_to_app_page(username)
            except firebase_admin.auth.UserNotFoundError:
                st.sidebar.error(f'E-mailadres of wachtwoord is onjuist', icon='⚠️')

    load_welcome()


if st.session_state.logged_in:
    run_app(st.session_state.get('username'))
    if st.sidebar.button('Log out'):
        switch_to_login_page()  # Switch back to the login page on logout
else:
    show_auth()
