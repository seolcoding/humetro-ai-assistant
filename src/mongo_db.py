import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

MONGODB_PW = os.getenv('MONGODB_PW')
uri = f"mongodb+srv://snowdelver:{MONGODB_PW}@cluster0.ivc4bze.mongodb.net/?retryWrites=true&w=majority"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)


def save_to_mongo(user_input, bot_response):
    db = client.get_database('haa')
    records = db.chatbot
    records.insert_one({'user_input': user_input,
                        'bot_response': bot_response,
                        'timestamp': datetime.now()})
