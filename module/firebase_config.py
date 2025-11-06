
import pyrebase

firebaseConfig = {
    "apiKey": "AIzaSyB92lB8qxZJjFfsB5rOPugnUAgflEM4MlE",
    "authDomain": "speechperfect-b00c7.firebaseapp.com",
    "databaseURL": "https://speechperfect-b00c7-default-rtdb.firebaseio.com/",
    "projectId": "speechperfect-b00c7",
    "storageBucket": "speechperfect-b00c7.appspot.com",
    # "messagingSenderId": "YOUR_SENDER_ID",
    # "appId": "YOUR_APP_ID"
}

firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()
db = firebase.database()

# --- Hardcoded Authentication ---
# (Make sure this user exists in Firebase Authentication > Users)
EMAIL = "wereen0909@gmail.com"
PASSWORD = "Wereenz0909?"

try:
    user = auth.sign_in_with_email_and_password(EMAIL, PASSWORD)
    print("✅ Firebase authentication successful!")
except Exception as e:
    print("❌ Firebase authentication failed:", e)
    user = None

