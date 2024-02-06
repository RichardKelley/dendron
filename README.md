# 🌳 Dendron

For a full installation that includes supports 🤗 Transformers out of the box, run

```
pip install dendron[full]
```

This will install transformers, bitsandbytes, accelerate, and sentencepiece, as well as protobuf and pytest. If you do this, you should look at installing and using (Flash Attention)[https://github.com/Dao-AILab/flash-attention], which is just a pip install, but has prerequisites that you should manually check. It's worth it though - maybe doubling your inference speeds. 

For a minimal installation that only supports basic behavior tree constructs, you can just do 

```
pip install dendron
```

This will give you the basic control structures needed for behavior trees, but none of the dependencies for running LLM nodes out of the box. You can always install these later as needed.

### Behavior Trees for Structured Programming of LLMs

[TODO]

## Examples

[TODO]

### Chat
### 20 Questions/Animal-Vegetable-Mineral
### Document Analysis
### Document Search
### Physical Inspection (digital twins)
### Embodied Agents

## Documentation

### Tutorials

[TODO]

### How-to Guides

[TODO]

### Explanations

[TODO]

## Acknowledgements

This work was supported in part by the Federal Transit Administration and Regional Transportation Commission of Washoe County.