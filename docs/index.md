---
title: Dendron
---

# 🌳 Dendron

Dendron is a library for building applications using behavior trees and large language models. 

For installation instructions [click here](install.md).

For a tutorial overview of the library, [click here](tutorial_intro.md). In the tutorial, you will use Dendron to build a chat system that:

1. Performs speech recognition from a microphone using a transformer-based ASR model.
2. Handles chat template formatting for a quantized LLM.
3. Generates speech from text using a transformed-based TTS model. 
4. Intelligently chunks the chat output using rule-based AI to prevent overwhelming the TTS model.
5. Uses another language model to classify user input to determine when it is socially appropriate for the language model to end the conversation.

The resulting behavior tree uses LLMs in 5 different ways, but can still be run on a **single RTX 3090**.

For an API overview, see the menu above.