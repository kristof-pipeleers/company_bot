import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore, auth, storage
import company_search_bot
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import extra_streamlit_components as stx
import json
import uuid
import jwt
import datetime
import secrets

# Initialize CookieManager
def get_manager():
    return stx.CookieManager()

cookie_manager = get_manager()

# Function to generate JWT token with additional information
def generate_jwt_token(user):
    payload = {
        'doc_id': user.uid,
        'email': user.email,
        'username': user.display_name,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    }
    return jwt.encode(payload, st.secrets["SECRET_JWT"], algorithm='HS256')

# Function to verify JWT token
def verify_jwt_token(token):
    try:
        payload = jwt.decode(token, st.secrets["SECRET_JWT"], algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None  # Token has expired
    except jwt.InvalidTokenError:
        return None  # Token is invalid

# When setting login status, generate and set a JWT token as a cookie
def set_login_status(user):
    jwt_token = generate_jwt_token(user)
    cookie_manager.set(cookie="jwt_token", val=jwt_token)

# When checking login status, use JWT token from the cookie
def check_login_status():
    jwt_token = cookie_manager.get(cookie="jwt_token")
    print(jwt_token)
    if jwt_token:
        decoded_token = verify_jwt_token(jwt_token)
        if decoded_token:
            return decoded_token
    return None

# When logging out, clear the JWT token from the cookie
def clear_login_status():
    cookie_manager.delete("jwt_token")

# Function to load CSS from a file and inject it into the app
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css('styles.css')


if not firebase_admin._apps:
    # Load service account credentials from secrets
    service_account_info = st.secrets["service_account"]

    # Initialize Firebase app with service account credentials
    cred = credentials.Certificate(json.loads(service_account_info))
    firebase_admin.initialize_app(cred, {'storageBucket': 'socs-415712.appspot.com'})
else:
    firebase_admin.get_app(name='[DEFAULT]')

# Firestore Database
db = firestore.client()

# Firebase Storage
bucket = storage.bucket()

# Constants
MIN_PASSWORD_LENGTH = 6   

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
        - 📡 **Externe data bronnen:** De chatbot maakt gebruik van meerdere databronnen en combineert de resultaten tot het optimale antwoord.
        
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

def send_custom_email(email, username, link):
    # Simple email text
    email_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    body {{
        font-family: Arial, sans-serif;
        line-height: 1.6;
    }}
    .content {{
        max-width: 600px;
        margin: 20px auto;
        padding: 20px;
        background: #f4f4f4;
        border-radius: 8px;
    }}
    .button {{
        display: inline-block;
        padding: 10px 20px;
        background: #3498db;
        color: #ffffff;
        text-decoration: none;
        border-radius: 5px;
        margin-top: 20px;
    }}
    </style>
    </head>
    <body>
    <div class="content">
        <h1>👋 Hi, {username}!</h1>
        <p>Verifieer uw e-mailadres om advAICE te kunnen gebruiken.</p>
        <a href="{link}" class="button">E-mail verifiëren</a>
        <p>Als u dit niet heeft aangevraagd, negeer dan deze e-mail.</p>
        <p>Bedankt,<br>Uw 🍃 advAICE Team</p>
    </div>
    </body>
    </html>
    """

    message = Mail(
        from_email='pkristof49@gmail.com',
        to_emails=email,
        subject='Verify your email for advAICE',
        html_content=email_body)
    try:
        sg = SendGridAPIClient(st.secrets['SENDGRID_KEY'])
        response = sg.send(message)
        print(response.status_code)
        print(response.body)
        print(response.headers)
    except Exception as e:
        print(e.message)


# Authentication Block
def show_auth():
    st.sidebar.title("🍃 advAI:green[CE]")
    choice = st.sidebar.radio('Login / Registreer', ['Login', 'Sign up'])

    email = st.sidebar.text_input('E-mailadres', placeholder="example@random.com")
    password = st.sidebar.text_input('Wachtwoord', type='password', help=f'Het wachtwoord moet minstens {MIN_PASSWORD_LENGTH} tekens lang zijn.', placeholder='Voer uw wachtwoord in')

    if choice == 'Sign up':
        username = st.sidebar.text_input('Voer de gebruikersnaam van je app in', placeholder="Jan Jansen")
        submit = st.sidebar.button('Mijn account aanmaken')
        if submit:
            if len(password) < MIN_PASSWORD_LENGTH:
                st.sidebar.error(f'Het wachtwoord moet minstens {MIN_PASSWORD_LENGTH} tekens lang zijn.', icon='⚠️')
            else:
                try:
                    user = auth.create_user(email=email, password=password, display_name=username)
                    # Send verification email
                    verification_link = auth.generate_email_verification_link(email, action_code_settings=None)
                    send_custom_email(email=email, username=username, link=verification_link)
                    st.sidebar.success('Je account is succesvol aangemaakt! Controleer je inbox om je e-mail te verifiëren.', icon='✅')
                    st.balloons()
                    
                except Exception as e:
                    st.sidebar.error(f'Fout bij het aanmaken van account: {e}', icon='⚠️')

    elif choice == 'Login':
        if st.sidebar.button('Login'):
            try:
                user = auth.get_user_by_email(email)
                # Check if email is verified
                if user.email_verified:

                    set_login_status(user)
                    # Add user data to Firestore
                    user_data = {'username': user.display_name, 'email': user.email, 'email_verified': user.email_verified}
                    db.collection('users').document(user.uid).set(user_data)
                    
                    # User data fetch from Firestore is assumed to be done here
                    st.sidebar.success(f'Succesvol ingelogd', icon='✅')
                    st.rerun()
                else:
                    st.sidebar.warning('Verifieer je e-mailadres om in te loggen.', icon='📧')
            except firebase_admin.auth.UserNotFoundError:
                st.sidebar.error(f'E-mailadres of wachtwoord is onjuist', icon='⚠️')
            except Exception as e:
                st.sidebar.error(f'Inlogfout: {e}', icon='⚠️')

    load_welcome()


decoded_token = check_login_status()
if decoded_token:
    company_search_bot.run_app(db=db)
    if st.sidebar.button('Log out'):
        clear_login_status()
        st.rerun()
else:
    show_auth()

