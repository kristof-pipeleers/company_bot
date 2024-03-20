import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore, auth, storage
import LLM_processes
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from jinja2 import Environment, FileSystemLoader
import json
import jwt
import datetime

file_loader = FileSystemLoader('.')
env = Environment(loader=file_loader)
template = env.get_template('welcome_message.jinja2')
welcome_message = template.render()

SECRET_KEY = st.secrets["SECRET_JWT"]

def create_jwt_token(user):
    """Create a JWT token with an expiration time."""
    expiration = datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    payload = {
        'uid': user.uid,
        'exp': expiration
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return token

def verify_jwt_token(token):
    """Verify the JWT token is valid and not expired."""
    try:
        # Decode the token
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload.get('uid')  # Return the payload if the token is valid
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def set_login_status(user):
    """Store JWT token in the session state upon successful login."""
    token = create_jwt_token(user)
    current_params = st.query_params
    current_params['session_token'] = token

def clear_login_status():
    """Clear the login session."""
    st.query_params.clear()

def check_login_status():
    """Check if the user is logged in by verifying the JWT token."""
    current_params = st.query_params
    token = current_params.get('session_token')
    if token:
        return verify_jwt_token(token)
    else:
        return None

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
    st.markdown(welcome_message, unsafe_allow_html=True)
    
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


uid = check_login_status()
print(uid)
if uid:
    LLM_processes.run_app(db=db, uid=uid)
    if st.sidebar.button('Log out'):
        clear_login_status()
        st.rerun()
else:
    show_auth()

