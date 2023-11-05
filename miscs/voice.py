from dotenv import load_dotenv
from audiorecorder import audiorecorder, AudioSegment
from gtts import gTT


def TTS(response):
    filename = "output.mp3"
    tts = gTTS(text=response, lang='ko')
    tts.save(filename)

    with open(filename, 'rb') as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio autoplay="True">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
        """
        st.markdown(md, unsafe_allow_html=True)
    os.remove(filename)


def ask_gpt(messages, model):
    response = openai.ChatCompletion.create(model=model, messages=messages)
    print(response)
    system_message = response["choices"][0]['message']
    print(system_message['content'])
    return system_message['content']


def STT(audio: AudioSegment):
    filename = "input.mp3"
    audio.export(filename, format="mp3")

    audio_file = open(filename, 'rb')
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    audio_file.close()
    os.remove(filename)
    return transcript['text']
