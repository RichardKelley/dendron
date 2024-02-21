---
title: Tutorial Introduction
---

# Building a Local Chat Agent with Dendron

Dendron is a Python library for building applications using behavior trees to structure and coordinate the execution of one or more large transformer-based neural networks. I'm going to assume that you have some idea of what transformer-based models are, and in this first tutorial show you how to structure the use of those models using behavior trees. So that we are all starting from a similar place, let's begin by giving a high-level description of how behavior trees work before we dive into implementing one using Dendron.

## A Quick Overview of Behavior Trees

A **behavior** is a discrete unit of action in the world. If this definition feels ambiguous, that's because it is: in some applications an individual behavior is a coarse thing; in others a very fine unit of execution. You'll need to understand your individual problem and solution spaces to decide what the natural behaviors are in your case. To solve most problems requires more than one behavior: a sequence of actions, or perhaps an attempt at multiple possibile actions until one succeeds, or something still more complicated. Once you have multiple behaviors, you have to decide how to _structure their interactions_. A **behavior tree** is a particular framework for thinking about how behaviors should interact.

In a behavior tree, we represent a collection of behaviors as a _tree_, in which individual behaviors are the leaf nodes and the interior nodes of the tree contain the logic that coordinates the behaviors. To make the tree do something, we **tick** the root node. The root then propagates that tick down the tree. Each node in the tree knows what to do when it is ticked. Once a node is done performing the actions associated with its tick operation, it returns a status back to its parent: either `SUCCESS` or `FAILURE`, or less commonly `RUNNING` to indicate that the node is still going. A tick flows through the tree according to the logic implemented by the interior nodes of the tree.

An example of a behavior tree (in fact, the behavior tree that you will have built by the end of this tutorial) can be seen here:

<center>
<markdown figure>
![image](img/4_asr_voice_chat.svg)
</figure>
</center>

This tree uses Dendron to implement a chat agent that listens to a human via a microphone, transcribes speech using a transformer-based model, generates responses to the human via a large language model, and replies using text-to-speech via a third transformer-based model. Using still another model, the tree also analyzes the human's input and determines, based on the human's words, if it is time for the agent to say goodbye. If you are looking at that tree and thinking that it looks a little backwards, good! Read on: you'll see why it is structured the way it is in Part 2 of the tutorial. The tree and its notation will be fully explained by Part 4 of the tutorial.

## Getting Started and Moving Forward

I'm going to assume that you have installed Dendron and its requirements into your Python environment. If you haven't yet, check out [this link](install.md) or run

```bash
pip install dendron
```

in your Python environment.

Once you have the system installed, you can move on to building [a behavior tree with a single node](0_tutorial_single_node.md).

