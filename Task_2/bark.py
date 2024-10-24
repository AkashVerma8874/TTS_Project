import os
os.environ["SUNO_USE_SMALL_MODELS"] = "True"
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio

# download and load all models
preload_models()

# generate audio from text
text_prompt = """
     गाँवों में लोग सामुदायिक जीवन जीते हैं, जहाँ पड़ोसी एक-दूसरे के सुख-दुख में भागीदार होते हैं। यहाँ परंपराएँ और रीति-रिवाज पीढ़ियों से चली आ रही हैं। लोग एक साथ त्योहार मनाते 
"""
audio_array = generate_audio(text_prompt)

# save audio to disk
write_wav("bark_generation.wav", SAMPLE_RATE, audio_array)
  
# play text in notebook
Audio(audio_array, rate=SAMPLE_RATE)

