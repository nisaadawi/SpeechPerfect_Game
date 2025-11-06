# game/firebase_fetcher.py
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
db = firebase.database()

def get_emotion():
    try:
        value = db.child("emotion").get()
        if value.val():
            return value.val()
        else:
            return "Neutral"
    except Exception as e:
        print("❌ Firebase fetch failed:", e)
        return "Neutral"
