
# Speech-to-Text System Using SpeechT5 Model






## 1 Introduction
This report presents the implementation of a speech synthesis system using the SpeechT5Processor, SpeechT5ForTextToSpeech, and SpeechT5HifiGan classes from the Hugging Face Transformers library. The objective of the project is to convert text into natural-sounding speech using pretrained models for both the text-to-speech transformation and vocoding.
##  2. Model Overview
The system is built using the following components:

1. SpeechT5Processor: A tokenizer and feature extractor for handling text input and processing it into tensors that can be passed into the text-to-speech (TTS) model.
2. SpeechT5ForTextToSpeech: A pretrained Transformer-based model that converts tokenized text inputs into mel-spectrogram representations of speech.
3. SpeechT5HifiGan: A neural vocoder based on the HiFi-GAN architecture that converts the mel-spectrogram produced by the TTS model into a waveform.

These components work together to synthesize speech from text.
## 3. Dataset Description
For this experiment, an external dataset was used to provide the x-vector embeddings representing speaker-specific voice characteristics:

1. Dataset: Matthijs/cmu-arctic-xvectors

2. Description: This dataset contains speaker embeddings that can be used to adjust the synthesized voice’s characteristics (e.g., tone, pitch, and speaking style). Each embedding is a vector representation of a speaker's voice.

3. Data Split: validation

4. Data Format: The dataset consists of speaker embeddings in vector format, which are used as inputs to condition the speech synthesis model.
The specific entry used from this dataset is at index 7306, which corresponds to the x-vector embedding used to simulate the speaker’s voice in this example.
## 4. Implementation Details
# Installation:
1. pip install transformers

2. pip install datasets
3. pip install torch
4. pip install soundfile
5. pip install numpy
6. pip install librosa
These packages form the core environment necessary to run and experiment with SpeechT5 for speech synthesis tasks.
# Code
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from datasets import load_dataset

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

inputs = processor(text="A village is a small settlement typically found in rural areas, smaller than a town but larger than a hamlet.", return_tensors="pt")

# load xvector containing speaker's voice characteristics from a dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

sf.write("speech.wav", speech.numpy(), samplerate=16000)
# Explanation:
1. Loading Pretrained Models:

a). The SpeechT5Processor is responsible for tokenizing the input text and preparing it for the TTS model.

b). The SpeechT5ForTextToSpeech is a transformer-based model that generates mel-spectrograms from the tokenized text.

c). The SpeechT5HifiGan vocoder is used to convert the generated mel-spectrogram into audio waveforms.

2. Text Input: The text, "A village is a small settlement typically found in rural areas, smaller than a town but larger than a hamlet.", is tokenized using the processor and converted into input tensors that can be passed to the model.

3. Speaker Embeddings: A pre-trained x-vector embedding is extracted from the CMU Arctic X-vectors dataset to provide the speaker characteristics. This allows the synthesized voice to mimic a specific speaker.

4. Speech Generation: The TTS model takes the text input and speaker embedding to generate a mel-spectrogram. The HiFi-GAN vocoder then converts this spectrogram into a waveform, which is saved as an audio file, speech.wav.

## 5. Training Logs
Since the models used are pretrained, there was no further fine-tuning or training conducted in this implementation. The models were directly used for inference on the given text and speaker embeddings.

However, during pre-training of the models (by Microsoft), the following types of data were likely used:

1. LJ Speech Dataset: This dataset contains audio of a single speaker reading passages and is often used for TTS training.

2. VCTK Corpus: A dataset with multiple speakers providing varied accents and pronunciations, suitable for building speaker-conditioned TTS systems.

Unfortunately, detailed training logs for the original models are not publicly available, but training typically involves minimizing the mean squared error (MSE) between predicted and ground truth mel-spectrograms, as well as adversarial training to improve the quality of the vocoded waveform.
## 6. Performance Evaluation
# Qualitative Evaluation:
The synthesized speech generated using the above system sounds natural and closely mimics human speech, especially when speaker embeddings are used to condition the output. The speech produced was clear and understandable, with distinct intonations that matched the characteristics of the speaker embedding provided.
# Quantitative Evaluation:
No quantitative evaluation was performed in this report since the models were not fine-tuned or re-trained. However, typical metrics used for TTS system evaluation include:
1. Mean Opinion Score (MOS): A subjective evaluation where human listeners rate the quality of the speech.

2. Mel-Cepstral Distortion (MCD): A measure of the difference between synthesized and real speech.

3. Word Error Rate (WER): Used to evaluate how understandable the synthesized speech is when transcribed by an ASR (automatic speech recognition) system.

## 7. Conclusion and Future Work
This implementation demonstrates the successful use of pretrained SpeechT5 models for text-to-speech synthesis, utilizing speaker embeddings to personalize the voice characteristics. Future work could involve:

1. Fine-tuning the model with domain-specific text data to improve contextual fluency.
2. Collecting MOS scores by having human listeners rate the quality of the synthesized speech.
3. Using custom speaker embeddings or additional datasets for more diverse voice synthesis.
## example
