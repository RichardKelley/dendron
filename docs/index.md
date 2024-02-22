---
title: Dendron
---

# 🌳 Dendron

Dendron is a library for building artificial intelligence applications using behavior trees and large language models. 

In the last few years, behavior trees have become popular in game AI development and robotics. This has happened because behavior trees make it easy to compose simple behaviors (pieces of code) into collections of complex behaviors. Behavior trees also naturally enable a high degree of code modularity and reusability that is beneficial for building AI systems that interact with the world in complex ways. On the other hand, large language models appear to be very powerful, but are not easily composable, either with other language models or classical software components. By making it easy to build behavior trees that integrate large language models, Dendron gives you the ability to easily build, use, and reuse sophisticated artificial intelligence programs.

!!! info

    Never worked with behavior trees? We define them from scratch in [Tutorial 1](tutorial_intro.md#a-quick-overview-of-behavior-trees).

## Installation

You can use `pip` to install Dendron. For installation instructions, [click here](install.md).

## The Tutorial

For a tutorial overview of the library, [click here](tutorial_intro.md). In the tutorial, you will use Dendron to build a chat agent that:

1. Performs speech recognition from a microphone using a transformer-based ASR model.
2. Handles chat template formatting for a quantized LLM.
3. Generates speech from text using a transformer-based TTS model. 
4. Intelligently segments the chat output using rule-based AI to prevent overwhelming the TTS model.
5. Uses another language model to classify user input to determine when it is socially appropriate for the language model to end the conversation.

The resulting behavior tree uses **four different LLMs in five different ways**, but can still be run on a **single RTX 3090**.

## API Reference

For an API reference, see the menu to the left on this page.