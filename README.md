Bark=>
  Bark is a transformer-based text-to-audio model created by Suno. Bark can generate highly realistic, multilingual speech as well as other audio - including music, background noise and simple sound effects. The model can also produce nonverbal communications like laughing, sighing and crying. To support the research community, we are providing access to pretrained model checkpoints, which are ready for inference and available for commercial use.
 Quick Index
💻 Installation
🐍 Usage
💻 Installation=>
pip install git+https://github.com/suno-ai/bark.git
or

git clone https://github.com/suno-ai/bark
cd bark && pip install . 
🤗 Transformers Usage
Bark is available in the 🤗 Transformers library from version 4.31.0 onwards, requiring minimal dependencies and additional packages. Steps to get started:

First install the 🤗 Transformers library from main:
pip install git+https://github.com/huggingface/transformers.git
Run the following Python code to generate speech samples:
from transformers import AutoProcessor, BarkModel

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")

voice_preset = "v2/en_speaker_6"

inputs = processor("Hello, my dog is cute", voice_preset=voice_preset)

audio_array = model.generate(**inputs)
audio_array = audio_array.cpu().numpy().squeeze()
Listen to the audio samples either in an ipynb notebook:
from IPython.display import Audio

sample_rate = model.generation_config.sample_rate
Audio(audio_array, rate=sample_rate)
Or save them as a .wav file using a third-party library, e.g. scipy:

import scipy

sample_rate = model.generation_config.sample_rate
scipy.io.wavfile.write("bark_out.wav", rate=sample_rate, data=audio_array)
Usage in Python
🪑 Basics
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio

# download and load all models
preload_models()

# generate audio from text
text_prompt = """
      गाँवों में लोग सामुदायिक जीवन जीते हैं, जहाँ पड़ोसी एक-दूसरे के सुख-दुख में भागीदार होते हैं। यहाँ परंपराएँ और रीति-रिवाज पीढ़ियों से चली आ रही हैं। लोग एक साथ त्योहार मनाते .
"""
audio_array = generate_audio(text_prompt)

# save audio to disk
write_wav("bark_generation.wav", SAMPLE_RATE, audio_array)
  
# play text in notebook
Audio(audio_array, rate=SAMPLE_RATE)
