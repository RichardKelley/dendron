---
title: Dendron
---

# 🌳 Dendron

Dendron is a library for building applications using behavior trees and large language models. 

In the last few years, behavior trees have become popular in game AI development and robotics. This is because behavior trees

For installation instructions [click here](install.md).

For a tutorial overview of the library, [click here](tutorial_intro.md). In the tutorial, you will use Dendron to build a chat system that:

1. Performs speech recognition from a microphone using a transformer-based ASR model.
2. Handles chat template formatting for a quantized LLM.
3. Generates speech from text using a transformer-based TTS model. 
4. Intelligently chunks the chat output using rule-based AI to prevent overwhelming the TTS model.
5. Uses another language model to classify user input to determine when it is socially appropriate for the language model to end the conversation.

The resulting behavior tree uses **four different LLMs in five different ways**, but can still be run on a **single RTX 3090**.

For an API reference, see the menu to the left.