# 🌳 Dendron

<div align="center" style="margin-bottom: 0em;">

[![][arxiv-badge]][arxiv] [![][discord-badge]][discord] [![][twitter-badge]][twitter]


**Dendron is a library for building software agents using behavior trees and language models.**

</div>

### News

**April 2024:** New on arXiv: [Behavior Trees Enable Structured Programming of Language Model Agents
](https://arxiv.org/abs/2404.07439)

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

## The Paper

If you use Dendron in academic research, please cite the [paper](https://arxiv.org/abs/2404.07439):

```
@misc{kelley2024behavior,
      title={Behavior Trees Enable Structured Programming of Language Model Agents}, 
      author={Richard Kelley},
      year={2024},
      eprint={2404.07439},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```

## Acknowledgements

This work was supported in part by the Federal Transit Administration and the Regional Transportation Commission of Washoe County.

[arxiv-badge]: https://img.shields.io/badge/arXiv-2404.07439-B31B1B?style=flat-square&logo=arXiv&link=https%3A%2F%2Farxiv.org%2Fabs%2F2404.07439
[arxiv]: https://arxiv.org/abs/2404.07439

[discord]: https://discord.gg/ncBeGQJ9Bk
[discord-badge]: https://img.shields.io/badge/Discord-chat-%235865F2?logo=discord&logoColor=white&link=https%3A%2F%2Fdiscord.gg%2FncBeGQJ9Bk



[twitter]: https://twitter.com/richardkelley
[twitter-badge]: https://img.shields.io/twitter/follow/richardkelley