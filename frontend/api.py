import openai
import json

def langchain_response( model="chagpt-3.5-turbo", messages = "",stream=True):
    return openai.ChatCompletion.create(model=model, messages=messages)

def openai_response(model = "chatgpt-3.5-turbo", messages="", stream=True):
    result = openai.ChatCompletion.create(model=model, messages=messages)
    print('result', result)
    return json.loads(result)
    
