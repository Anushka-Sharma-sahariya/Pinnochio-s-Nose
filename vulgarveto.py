import streamlit as st
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from tempfile import NamedTemporaryFile
import os
import io
import pyaudio
import IPython.display as ipd
import time

def transcribe_chunk(audio_chunk):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_chunk) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return "Could not request results; {0}".format(e)

def split_audio(audio_file, chunk_duration_ms):
    audio = AudioSegment.from_wav(audio_file)
    chunk_count = len(audio) // chunk_duration_ms + 1
    chunks = []
    for i in range(chunk_count):
        start_time = i * chunk_duration_ms
        end_time = (i + 1) * chunk_duration_ms
        if end_time > len(audio):
            end_time = len(audio)
        chunk = audio[start_time:end_time]
        with NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_filename = temp_file.name
            chunk.export(temp_filename, format="wav")
            chunks.append(temp_filename)
    return chunks

def transcribe_audio(audio_file, chunk_duration_ms):
    audio_chunks = split_audio(audio_file, chunk_duration_ms)
    transcriptions = []
    for chunk in audio_chunks:
        transcription = transcribe_chunk(chunk)
        transcriptions.append(transcription)

    full_text = ' '.join(transcriptions)
    return full_text

def filter_bad_words(text, bad_words, beep_sound):
    words = text.split()
    filtered_words = []
    for word in words:
        if word.lower() in bad_words:
            filtered_words.append(beep_sound)
        else:
            filtered_words.append(word)
    filtered_text = ' '.join(filtered_words)
    return filtered_text

# def record_audio(duration):
#     frames = []
#     p = pyaudio.PyAudio()
#     CHUNK = 1024
#     FORMAT = pyaudio.paInt16
#     CHANNELS = 1
#     RATE = 44100
#     RECORD_SECONDS = duration
#     COUNTDOWN_SECONDS = 5

#     stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
#     print("Recording...")

#     for i in range(COUNTDOWN_SECONDS, 0, -1):
#         print(f"Recording starts in {i} second(s)...")
#         time.sleep(1)

#     for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
#         data = stream.read(CHUNK)
#         frames.append(data)

#     print("Finished recording.")
#     stream.stop_stream()
#     stream.close()
#     p.terminate()

#     # Save the recorded audio as a temporary .wav file
#     temp_audio_file = NamedTemporaryFile(delete=False, suffix=".wav")
#     temp_audio_filename = temp_audio_file.name
#     with open(temp_audio_filename, "wb") as f:
#         for frame in frames:
#             f.write(frame)

#     return temp_audio_filename


# def record_audio(duration):
#     frames = []
#     p = pyaudio.PyAudio()
#     CHUNK = 1024
#     FORMAT = pyaudio.paInt16
#     CHANNELS = 1
#     RATE = 44100
#     RECORD_SECONDS = duration
#     COUNTDOWN_SECONDS = 5

#     stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
#     print("Recording...")

#     for i in range(COUNTDOWN_SECONDS, 0, -1):
#         print(f"Recording starts in {i} second(s)...")
#         time.sleep(1)

#     for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
#         data = stream.read(CHUNK)
#         frames.append(data)

#     print("Finished recording.")
#     stream.stop_stream()
#     stream.close()
#     p.terminate()
    
#      # Save the recorded audio as a temporary .wav file
#     temp_audio_file = NamedTemporaryFile(delete=False, suffix=".wav")
#     temp_audio_filename = temp_audio_file.name
#     with open(temp_audio_filename, "wb") as f:
#         for frame in frames:
#             f.write(frame)

#     return temp_audio_filename

def text_to_speech(text, language='en'):
    tts = gTTS(text=text, lang=language, slow=False)
    audio_file = io.BytesIO()
    tts.write_to_fp(audio_file)
    audio_file.seek(0)
    return audio_file

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("https://media.licdn.com/dms/image/D4D12AQFbZfiSO7Hzfg/article-cover_image-shrink_720_1280/0/1697613632807?e=1718841600&v=beta&t=HcrHz0AExQDccQy8bh6BnhBI3CsPqAUnaCEMAW5Hm5s");
        background-attachment: fixed;
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
)
def main():
    bad_words_file_path = "dataset.txt"
    with open(bad_words_file_path, 'r') as file:
        bad_words = file.read().splitlines()
    beep_sound = "BEEP"

    st.title("VulgarVeto")

    # File upload section
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

    if uploaded_file:
        # Display the audio file
        st.audio(uploaded_file)

        transcription = ""
        if st.button("Filter Bad Words"):
            transcription = transcribe_audio(uploaded_file, 15000)
            st.write("Transcription:")
            st.write(transcription)
            filtered_text = filter_bad_words(transcription, bad_words, beep_sound)
            st.write("Filtered Transcription:")
            st.write(filtered_text)

            # Convert filtered text to speech
            filtered_audio = text_to_speech(filtered_text)
            st.audio(filtered_audio, format='audio/wav')

# def main():
#     bad_words_file_path = "dataset.txt"
#     with open(bad_words_file_path, 'r') as file:
#         bad_words = file.read().splitlines()
#     beep_sound = "BEEP"
#     st.title("VulgarVeto")

#     # Add a radio button to choose between uploading or recording audio
#     audio_source = st.radio("Select audio source", ["Upload audio file", "Record audio"])

#     if audio_source == "Upload audio file":
#         # File upload section
#         uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])
#         if uploaded_file:
#             # Display the audio file
#             st.audio(uploaded_file)
#             if st.button("Filter Bad Words"):
#                 transcription = transcribe_audio(uploaded_file, 15000)
#                 st.write("Transcription:")
#                 st.write(transcription)
#                 filtered_text = filter_bad_words(transcription, bad_words, beep_sound)
#                 st.write("Filtered Transcription:")
#                 st.write(filtered_text)
#                 # Convert filtered text to speech
#                 filtered_audio = text_to_speech(filtered_text)
#                 st.audio(filtered_audio, format='audio/wav')

#     elif audio_source == "Record audio":
#         # Add a button to start recording
#         if st.button("Start Recording"):
#             # Record audio for 5 seconds
#             recorded_audio = record_audio(5)
            
#             # Save the recorded audio to a temporary file
#             with NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
#                 temp_filename = temp_file.name
#                 with open(temp_filename, 'wb') as f:
#                     f.write(recorded_audio)

#             # Display the recorded audio
#             st.audio(recorded_audio, format='audio/wav')

#             if st.button("Filter Bad Words"):
#                 with open(temp_filename, 'rb') as f:
#                     recorded_audio_file = f.read()
#                 transcription = transcribe_audio(io.BytesIO(recorded_audio_file), 15000)
#                 st.write("Transcription:")
#                 st.write(transcription)
#                 filtered_text = filter_bad_words(transcription, bad_words, beep_sound)
#                 st.write("Filtered Transcription:")
#                 st.write(filtered_text)
#                 # Convert filtered text to speech
#                 filtered_audio = text_to_speech(filtered_text)
#                 st.audio(filtered_audio, format='audio/wav')

if __name__ == "__main__":
    main()
