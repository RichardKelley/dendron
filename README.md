# 🌳 Dendron

### Behavior Trees for Structured Programming of LLMs

Behavior trees are a technique for building complex reactive agents by composing simpler behaviors in a principled way. The behavior tree abstraction arose from Robotics and Game AI, but the premise of Dendron is that this abstraction can enable more sophisticated language-based agents. 

Here is an example behavior tree that implements a chat agent. This agent listens to a human via microphone, performs automatic speech recognition (ASR), uses a chat model to generate a response, and plays the audio of that response using a text-to-speech (TTS) system. All locally, using models downloaded from Hugging Face:

![image](https://github.com/RichardKelley/dendron/raw/main/docs/img/4_asr_voice_chat.svg)

You can build this agent by following [the tutorial here](https://richardkelley.io/dendron/tutorial_intro).

### Installation

To install Dendron, run

```
pip install dendron
```

This will automatically install torch, transformers, bitsandbytes, accelerate, and sentencepiece, and protobuf. You should consider installing and using [Flash Attention](https://github.com/Dao-AILab/flash-attention), which is just a pip install, but has prerequisites that you should manually check. It's worth it though - maybe doubling your inference speeds. 

## Examples

For examples of basic language model node usage, see the example notebooks in this repository. For larger and more interesting examples, see the [examples repo](https://github.com/RichardKelley/dendron-examples).

## Documentation

You can find the main documentation for Dendron [here](https://richardkelley.io/dendron/). This includes a full tutorial building a chat agent that has text-to-speech and automatic speech recognition capabilities, and an API reference.

## Acknowledgements

This work was supported in part by the Federal Transit Administration and the Regional Transportation Commission of Washoe County.