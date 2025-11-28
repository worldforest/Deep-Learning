from openai import OpenAI

client = OpenAI(api_key='')


response = client.audio.speech.create(model='tts-1', voice='coral', input="겁나 졸려 졸려 졸려 졸려 졸려")
response.stream_to_file("./voice2.mp3")
