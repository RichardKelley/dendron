---
title: Dendron Tutorial Introduction
---

# Building a Local Chat Agent with Dendron

Dendron is a Python library for building applications using behavior trees to structure and coordinate the execution of one or more large transformer-based neural network models. I'm going to assume that you have some idea of what transformer-based neural nets are, and in this introductory tutorial I'll show you how to use behavior trees to do "structured programming" with those models. So that we are all starting from a similar place, let's begin by giving a high-level description of how behavior trees work before we dive into implementing one using Dendron.

## A Quick Overview of Behavior Trees

A **behavior** is a discrete unit of action performed in the world. If this definition feels ambiguous, that's because it is: in some applications an individual behavior is a coarse thing; in others a very fine unit of execution. Examples of behaviors are `transcribe a segment of audio data into English language text` or `Look at this image and output the string "yes" if it contains a puppy` or `generate a string of text based on the given chat history`. You'll need to understand your individual problem and solution spaces to decide what the natural behaviors are in your case. To solve most problems requires more than one behavior: a sequence of actions, or perhaps an attempt at multiple possible actions until one succeeds, or something still more complicated. Once you have multiple behaviors, you have to decide how to _structure their interactions_. A **behavior tree** is a particular framework for thinking about how behaviors should interact.

In a behavior tree, we represent a collection of behaviors as a _tree_, in which individual behaviors are the leaf nodes and the interior nodes of the tree contain the logic that coordinates the behaviors. To make the tree do something, we **tick** the root node. The root then propagates that tick down the tree. Each node in the tree knows what to do when it is ticked. Once a node is done performing the actions associated with its tick operation, it returns a status back to its parent: either `SUCCESS` or `FAILURE`, or less commonly `RUNNING` to indicate that the node is still going. The tick signal flows through the tree according to the logic implemented by the tree's interior nodes.

An example of a behavior tree (in fact, the behavior tree that you will have built by the end of this tutorial) can be seen below. You can click to zoom the image:

<center>
<markdown figure>
![image](img/4_asr_voice_chat.svg)
</figure>
</center>

This tree implements a chat agent that listens to a human via a microphone, transcribes the audio using a transformer-based speech recognition model, generates responses to the human via a large language model that has been tuned for chat, and replies using text-to-speech via a third transformer-based model. Using still another model, the tree also analyzes the human's input and determines, based on the human's words, if it is time for the agent to say goodbye and end the chat. 

!!! note 

    If you are reading the tree from left to right and thinking that it looks a little backwards, good! Read on: you'll see why it is arranged that way in Part 2 of the tutorial. The tree and its notation will be fully explained by the end of Part 4 of the tutorial.

## Getting Started and Moving Forward

I'm going to assume that you have installed Dendron and its requirements. If you haven't yet, run

```bash
pip install dendron
```

in your Python environment, or check out [this link](install.md) for details.

Once you have Dendron installed, you can move on to building [a behavior tree with a single node](0_tutorial_single_node.md).

