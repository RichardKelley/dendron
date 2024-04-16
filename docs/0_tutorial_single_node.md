---
title: Dendron Tutorial Part 0. A Single Node
---

# 0. Building a Chat System with Dendron: A Single Node

The real power of behavior trees comes from their ability to organize collections of behaviors that interact in complex ways, but to get started using a language model as quickly as possible let's build a tree with a single language model `TreeNode` that we can chat with. We'll have to manage the chat state ourselves, but in [part 2](2_tutorial_implicit_seq.md) we'll see how to make a more complex tree handle that state for us.

If you find this tutorial too verbose and you just want to get the code, you can find the notebook for this part [here](https://github.com/RichardKelley/dendron-examples/blob/main/tutorial_1/part_0.ipynb){:target="_blank"}.

## Causal Language Models in Dendron

We start by importing `dendron`, and then we import two classes that we can use to define a language model node:

```python linenums="1"
import dendron
from dendron.actions.causal_lm_action import CausalLMActionConfig, CausalLMAction
```

A `CausalLMAction` is an _action node_ that maintains a neural network and executes that network whenever it receives a _tick_ from its parent node in the tree. Recall from the [tutorial introduction](tutorial_intro.md) that the leaf nodes of a behavior tree are the nodes that actually "do stuff:" either they test predicates (_condition nodes_) or they perform tasks (_action nodes_). In this case, we want a behavior that takes in a chat history, runs an autoregressive language model on that history, and then prints the next reply from the model. The language model capabilities of Dendron are currently based on the Hugging Face 🤗 Transformers library, and `CausalLMAction` is the Dendron class that wraps Transformers' ability to load and run autoregressive models. The `CausalLMActionConfig` is a supporting dataclass that lets you specify options such as which model to download from the Hugging Face Hub, whether or not to quantize that model, how inputs are communicated to the model, how much text to generate, and so on. If you're curious about all of the options, look at [the documentation for the config](api/actions/causal_lm_action.md#dendron.actions.causal_lm_action.CausalLMActionConfig){:target="_blank"}.

Next we specify a configuration for our node:

```python linenums="1"
chat_behavior_cfg = CausalLMActionConfig(load_in_4bit=True,
                                         max_new_tokens=128,
                                         do_sample=True,
                                         top_p=0.95,
                                         use_flash_attn_2=True,
                                         model_name='openchat/openchat-3.5-0106')
```

!!! warning

    If you haven't installed flash attention, you will need to set `use_flash_attn_2 = False` in this and all of the other model configs of the tutorial.

There are few points to note about this configuration. First, we specify that we want to load our model using _4-bit quantization_. This means that we use less precision for each model weight, which leads to lower memory consumption. This is critical for running larger models on smaller GPUs. The `openchat_3.5` model we are using has billions of parameters, so it would be impossible to run without this quantization. Next, we specify that we are using Flash Attention 2. This makes inference substantially faster, but unfortunately isn't supported on older GPUs. The parameters `do_sample` and `top_p` specify a sampling strategy known as "nucleus sampling." You can read more about sampling strategies and approaches to generating text with language models [here](https://huggingface.co/blog/how-to-generate){:target="_blank"}.

!!! tip

    If you find that `openchat_3.5` is too big for you to run with its 7 billion parameters, there are smaller models you can explore that should work even on GPUs with less VRAM. Two that might be worth trying are `microsoft/phi-1_5` which has 1.3 billion parameters and `google/gemma-2b` which has 2 billion parameters. You can switch to these other models by changing the `model_name` parameter above. But note that to use `google/gemma-2b` you do have to agree to Google's terms first or else you'll get an error when you try to download the weights.

Next we create our node and our tree:

```python linenums="1"
chat_node = CausalLMAction('chat_node', chat_behavior_cfg)
tree = dendron.BehaviorTree("chat_tree", chat_node)
```

The first arguments to `CausalLMAction` and `BehaviorTree` are _names_. These can be anything you'd like and are useful for debugging and logging. We initialize the `chat_node` using the configuration we just created, and then we pass the resulting node to the `BehaviorTree` constructor as the second argument. In general, we initialize `BehaviorTree` instances by specifying a name and a root node. In this case, the root of the tree is the entire tree. 

!!! warning

    Running the `CasaulLMAction` constructor will automatically download the weights of the model you name in the config, unless you set `auto_load` to `False` in your `CausalLMActionConfig`. Remember that these models with their billions of parameters can take time and space to download and store.

You can visualize the behavior tree that results from the above as follows:

<center>
<markdown figure>
![image](img/0_single_node.svg){width="200"}
</figure>
</center>

Not much to admire quite yet; it would be just as easy to use the 🤗 Transformers library directly for a simple use case like this. But let's keep going and see what we can build on this foundation.

## Input and Output Processing

Next we need to decide how we want to manage the state of the chat. This ends up being a little annoying no matter what libraries you use, because every model has its own "chat template" for converting a structured chat history into a sequence of tokens. Here's an example chat using the format that `openchat_3.5` prefers:

```python linenums="1"
chat = [
    {"role": "GPT4 Correct User", "content": "Hello, how are you?"},
    {"role" : "GPT4 Correct Assistant", "content" : "I am well. How are you?"}
]
```

This is the format in which we'll track the chat state, so we'll need a way to convert between this form and a simple string that can be tokenized and passed to our model. This kind of transformation is common enough that Dendron lets you define custom _input processors_ and _output processors_ to implement the required transformations during every tick operation. It turns out that the tokenizer for `openchat_3.5` has built-in functionality to convert chat objects into strings, so we can write a simple input processor to perform that conversion:

```python linenums="1"
def chat_to_str(self, chat):
    return self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

tree.root.set_input_processor(chat_to_str)
```

We don't want to perform tokenization quite yet (that is done inside of `CausalLMAction`'s `tick` function), but we do want to add a _generation prompt_, which is a hint to the model that it should generate text in the persona of the assistant. If you don't do this, some models may generate text as if they were the user, which can be entertaining but is probably not what you're looking for.

In general, input processors for `CausalLMAction` should take `self` and a single argument that is a model input, and should return a single string output.

!!! warning

    Input and output processors (and the pre- and post-tick functions we'll meet later on) are added to the class as if they were originally part of the class definition. Keep this in mind if you are using a custom `TreeNode` class to maintain some private information, since added functions will have access to that information just as any other member function would. 

Converting from text strings to chat history is a bit more complicated:

```python linenums="1"
def str_to_chat(self, str):
    key = "GPT4 Correct Assistant:"
    idx = str.rfind(key)
    response = str[idx+len(key):]
    chat = self.blackboard[self.input_key]
    chat.append({"role" : "GPT4 Correct Assistant", "content" : response})
    return chat

tree.root.set_output_processor(str_to_chat)
```

During a `tick`, the model generates a sequence of integer tokens that are decoded back into a string. Here's an example of the decoded model output if we were to start with the `chat` list as above and have the human type "I am excited!" as the first response:

```
GPT4 Correct Gpt4 Correct User: Hello, how are you? GPT4 Correct Gpt4 Correct Assistant: I am well. How are you? GPT4 Correct User: I am excited! GPT4 Correct Assistant: That's great to hear! It's always nice to be excited about something. Is there anything specific that has you feeling this way?|
```

In order to continue the chat, we have to grab the agent's most recent reply and add it as the `"content"` value of a correctly formatted dictionary at the end of the list. That is precisely what the `str_to_chat` function does. In general, output processors for `CausalLMAction` should convert the decoded string output of a language model into whatever structured format is being used to track inputs (in this case the "list of dictionaries" format). 

With input and output processors in mind, we can visualize the flow of control inside the typical `tick` function as follows:

<center>
<markdown figure>
![image](img/input-tick-output.svg){:width="400px"}
</figure>
</center>

The output processor returns the chat history it constructs back to the `tick` function, which then writes that history to the behavior tree's _blackboard_.

## Blackboards

You probably noticed that `str_to_chat` makes reference to member variables called `self.blackboard` and `self.input_key`. One of the defining concepts of behavior trees is that nodes **only** communicate directly in two ways:

1. Parents `tick` their children.
2. Children reply to their parents by returning a `NodeStatus`. This status can be `SUCCESS`, `FAILURE`, or `RUNNING`.

These rules imply that some other mechanism is required if nodes need to share any other kind of information up or down the tree. The way that this is handled is via a `Blackboard` object. Every `BehaviorTree` has an associated `Blackboard`, and every `TreeNode` in the tree has access to that blackboard. You can think of a `Blackboard` as an in-memory key-value store that is shared by all of the nodes in a behavior tree. Keys are generally strings, but values can be any object that you can put into a Python `dict`. In the case of `CausalLMAction`, one of the optional arguments to `CausalLMActionConfig` is `input_key`, which specifies the key in the blackboard that will hold the prompt that the model should consume on the next `tick`. The value for `input_key` defaults to `"in"`, which is fine as long as you only have one language model in your tree. Otherwise you should choose a sensible name for your `input_key` (and probably for your `output_key` as well; see the [the documentation for CausalLMActionConfig](api/actions/causal_lm_action.md#dendron.actions.causal_lm_action.CausalLMActionConfig){:target="_blank"} for additional details).

## The Chat Loop

Our tree is all set up now, so all we have to do is set up a loop to chat with our agent:

```python linenums="1"
while True:
    input_str = input("Input: ")
    chat.append({"role": "GPT4 Correct User", "content" : input_str})
    tree.blackboard["in"] = chat

    tree.tick_once()

    print("Output: ", tree.blackboard["out"][-1]["content"])
    if "Goodbye" in tree.blackboard["out"][-1]["content"]:
        break
```

We loop forever, first getting a line of text from the standard input, then appending it to the `chat` list, and then writing the resulting list to `tree`'s `Blackboard` instance. Once the `Blackboard` is set up with the most recent human input, we call the tree's `tick_once` function, which works recursively: the tree ticks its root node, and then (in general) the root node propagates that tick down the tree according to the logic implemented by the tree's structure. In this instance there's only one node in the tree, so it gets ticked and that's it.

The `tick` function of the `CausalLMAction` node performs the following steps in order:

1. Retrieve a prompt from the node's `blackboard`, using the `input_key`.
2. Apply the input processor, if one exists.
3. Tokenize the prompt text.
4. Generate new tokens based on the prompt.
5. Decode the model output into a text string.
6. Apply the output processor, if one exists,
7. Write the result back to the `blackboard`, using the `output_key`.

As a result of this sequence of operations, when the `tick_once` call returns we can access the model's most recent output by getting the value stored at `tree.blackboard["out"][-1]`. We can then use the `"content"` key to get the string that the model produced. This is what we print before checking if we should `break` out of the loop: rather than continue forever, we examine the contents of the model's output string, and if the substring `"Goodbye"` appears then we `break` out of the loop, concluding the conversation.

If you want to see the chat history, you can examine the blackboard at a single key:

```python linenums="1"
print(tree.blackboard["out"])
```

or you can print the entire blackboard in tabular form:

```python linenums="1"
print(tree.blackboard)
```

There are some options for controlling how the blackboard prints. Depending on your needs you may find `Blackboard.set_print_len` helpful. See the [`Blackboard` documentation](api/blackboard.md#dendron.blackboard.Blackboard.set_print_len){: target="_blank"} for more information.

## Conclusion

If you followed the steps above, you should now have a model you can chat with! It's a little rough around the edges and doesn't really show off the power of behavior trees, so in the [next part](1_tutorial_seq.md) we'll make a _slightly_ more complex tree and add in the ability for our agent to speak out loud using a neural network TTS node that we'll write from scratch.