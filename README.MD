### Introduction

This Python program is a real-time version of this [Project](https://github.com/kelvi-ribeiro/auto-cut-ai), live demonstration of AI-powered video recognition capabilities. It integrates speech, gesture, and screen blink recognition to detect and display specific actions or keywords as they occur. Designed for practical and immediate feedback, the program captures live audio and video, processes it on the fly, and highlights detected cues such as special words, gestures, and screen blinks. This tool is ideal for showcasing the dynamic functionality of AI-driven video editing and recognition in an interactive and engaging way.

## Dependencies

- ![NumPy](https://img.shields.io/badge/NumPy-v1.24.3-blue)
- ![OpenCV](https://img.shields.io/badge/OpenCV-v4.7.0.72-red)
- ![MediaPipe](https://img.shields.io/badge/MediaPipe-v0.9.2.1-green)
- ![Whisper](https://img.shields.io/badge/Whisper-v1.0.0-purple)
- ![PyTorch](https://img.shields.io/badge/PyTorch-v2.0.0-black)
- ![SpeechRecognition](https://img.shields.io/badge/SpeechRecognition-v3.10.0-yellow)

## Running on Windows

1. Install [FFmpeg](https://www.wikihow.com/Install-FFmpeg-on-Windows)
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the project:
   ```
   python .\main.py
   ```

## Running on Linux

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run the project:
   ```
   python main.py
   ```