---
title: Chat with TTS and ASR
---


# 4. Building a Chat Agent with Dendron: Chat with TTS and ASR

In [Part 3](3_tutorial_llm_conditional.md) we learned how to add `CompletionCondition` nodes to our behavior tree, enabling our agent to analyze the human's inputs and terminate the conversation when appropriate. In this last part of the tutorial, we will make a small change in the code to use automatic speech recognition (ASR) to allow spoken human inputs to our agent instead of text. 

If you find this tutorial too verbose and you just want to get the code, you can find the notebook for this part [here](https://github.com/RichardKelley/dendron-examples/blob/main/tutorial_1/part_4.ipynb){:target="_blank"}.

## Simplifying Imports

Unlike the previous parts of this tutorial, we're going to start this part by showing you the tree you're going to build before we write any code:

<center>
<markdown figure>
![image](img/4_asr_voice_chat.svg)
</figure>
</center>

If you've worked through Part 3, just about all of this tree should look familiar. In fact, the only difference between this tree and the one from Part 3 is that we have replaced the `get_text_input` node with a new `get_voice_input` node. The rest of the tree is _identical_ to the previous one, and we can reuse all of the definitions we have already given. To that end, I have created a separate file named `dendron_tutorial_1.py` into which I have placed all of the class definitions we'll need for this part. You can see [the contents of that file here](4_supporting_code.md). With the old definitions tucked away in a supporting file, we can start building the final tree of this tutorial with a much smaller set of imports:

```python linenums="1"
from dendron_tutorial_1 import *

from whisper_mic import WhisperMic
```

## Getting Voice Inputs with Whisper Mic

We want an easy way to get access to the microphone, and the `whisper_mic` package provides exactly that. Once installed, you can import a single Python object `WhisperMic` that accesses your microphone and performs ASR.

!!! info

    Whisper is a neural net [released](https://openai.com/research/whisper){:target="_blank"} by OpenAI to perform speech recognition. It is quite good, but the package that OpenAI released focuses on supporting file-based transcription. The [Whisper Mic](https://github.com/mallorbc/whisper_mic){:target="_blank"} project makes it possible to transcribe speech directly from a microphone. You can install Whisper mic by running `pip install whisper-mic`.

Once we have access to a microphone and speech recognition, we need to wrap those capabilities into a custom Dendron node. The basic structure should be familiar by now:

```python linenums="1"
class GetVoiceInput(dendron.ActionNode):
    def __init__(self, latest_human_input_key = "latest_human_input"):
        super().__init__("get_voice_input")
        self.latest_human_input_key = latest_human_input_key
        self.mic = WhisperMic()

    def tick(self):
        t = np.arange(8000) / 16000
        t = t.reshape(-1, 1)
        beep = 0.2 * np.sin(2 * np.pi * 440 * t)
        sd.play(beep, 16000)
        
        self.blackboard[self.latest_human_input_key] = self.mic.listen()

        chat = self.blackboard["chat_history"]
        chat.append({"role": "GPT4 Correct User", "content" : self.blackboard[self.latest_human_input_key]})
        self.blackboard["in"] = chat
        
        return NodeStatus.SUCCESS        
```

In the `GetVoiceInput` constructor we initialize a `WhisperMic` object and keep track of the latest human input, just as in the previous part. The tick function can be understood in two parts:

1. Generate a beep to indicate when the human should speak.
2. Listen to the microphone and transcribe any human speech that is heard.

Lines 8-11 in `tick` play a beep for half a second (a satisfying 440Hz A note). We then use our `WhisperMic` object's `listen` member function, which returns transcribed text as a string. That string is then written to the blackboard. We go ahead and also update our chat object inside this `tick` and then return `SUCCESS`.

That's all there is to it. Obviously there's a lot of complexity hidden inside that `self.mic.listen()` call, since the library is creating an abstraction over both your hardware and the ASR process, but if all you need is a quick way to get transcribed speech from a mic, it's hard to beat line 13 above. Even better, `GetVoiceInput` is the only class we need to define. All that's left is to put the tree together.

## Building the Nodes of the Tree

Defining our tree nodes follows exactly the same process as in previous parts of this tutorial. Rather than rehash all of that, I'll show the code that we already know works, rearranged a bit since we've moved definitions into a separate file:

```python linenums="1"
speech_node = TTSAction("speech_node")
speech_node.add_post_tick(play_speech)

chat_behavior_cfg = CausalLMActionConfig(load_in_4bit=True,
                                         max_new_tokens=128,
                                         do_sample=True,
                                         top_p=0.95,
                                         use_flash_attn_2=True,
                                         model_name='openchat/openchat_3.5')

chat_node = CausalLMAction('chat_node', chat_behavior_cfg)

chat_node.set_input_processor(chat_to_str)
chat_node.set_output_processor(str_to_chat)
chat_node.add_post_tick(set_next_speech)

farewell_classifier_cfg = CompletionConditionConfig(
    input_key = "farewell_test_in",
    load_in_4bit=True,
    model_name='mlabonne/Monarch-7B',
    use_flash_attn_2=True
)

farewell_classification_node = CompletionCondition("farewell_classifier", farewell_classifier_cfg)
farewell_classification_node.add_pre_tick(farewell_pretick)
farewell_classification_node.add_post_tick(farewell_posttick)
```

Here we are defining the leaf nodes that rely on neural networks, following the steps in our previous tutorials. There are going to be a few nodes that we don't store into variables before insertion into the tree; those all rely on classes that we import from our definition file.

## Building the Tree

Once our leaf nodes are defined we can build the tree. Here's the code:

```python linenums="1"
speech_seq = Sequence("speech_seq", [
    MoreToSay(),
    speech_node
])

thought_seq = Sequence("thought_seq", [
    TimeToThink(),
    chat_node,
    SentenceSplitter()
])

goodbye_test = Sequence("goodbye_test", [
    farewell_classification_node,
    SayGoodbye(), 
    speech_node
])

conversation_turn = Fallback("conversation_turn", [
    speech_seq,
    thought_seq,
    GetVoiceInput()
])

root_node = Fallback("converse", [
    goodbye_test,
    conversation_turn,
])

tree = dendron.BehaviorTree("chat_tree", root_node)
```

We are following roughly the same sequence of steps as in previous parts of the tutorial, but here it should be clearer that we are building the tree from the leaves up. This is a good way to develop complex behavior trees: start with leaves for individual behaviors, combine them into subtrees, and then recursively combine the subtrees until you get a tree that does what you want. The only new thing we introduce is the use of `GetVoiceInput()` on line 21 in place of `GetTextInput()` from the previous parts of the tutorial. With that small change, you have enhanced your behavior tree into an agent that listens instead of reading. All that remains is to initialize the state of the tree and talk to it.

## Setting Up the Blackboard

State initialization is handled via the blackboard:

```python linenums="1"
tree.blackboard["chat_history"] = []
tree.blackboard["speech_in"] = []

tree.blackboard.register_entry(dendron.blackboard.BlackboardEntryMetadata(
    key = "latest_human_input",
    description = "The last thing the human said.",
    type_constructor = str
))
tree.blackboard["latest_human_input"] = None

tree.blackboard["completions_in"] = ["yes", "no"]
tree.blackboard["success_fn"] = farewell_success_fn
tree.blackboard["all_done"] = False
```

Nothing new here, just repeating what we've done in previous parts to get our blackboard set up.

## Running the Chat Loop

Last but not least, we can now run our chat loop:

```python linenums="1"
while not tree.blackboard["all_done"]:
    tree.tick_once()
```

Of course make sure you have a microphone turned on, or the system will just sit waiting to hear some speech. As soon as you run the loop you will hear a short beep - this is the indication that it's your turn to speak. Start talking, and when you finish you'll experience a short delay (a few seconds) and then the agent will respond. Once you're done, you can tell the agent "goodbye" or "farewell" or "peace out" (that last one should work just fine) and the conversation will end. 

## Conclusion

And so we come to the end of Tutorial 1. With consumer-grade hardware, you can carry on in conversation with your computer as long as you want. Admittedly, that probably won't be very long with this particular agent. In just a few minutes of playing with it, you can probably find a lot of rough edges. Some of these are fundamental, but many of them could be improved with a more sophisticated tree design. For example, if you forget to turn on your microphone until after you instantiate `GetVoiceInput`, the system will "listen," but will never hear you. It would be better for your tree to check for the microphone every few seconds. More annoying, the TTS is a little slow. If you're not very patient it can be hard to wait for answers or know when the agent is done speaking. You might try other models, or go with a completely non-neural TTS solution. That would be as easy as swapping out the `TTSAction` we have been using for one that is just _slightly_ different. You could also try adding in `print` statements to show the output of the `chat_node` before it gets sent to the `TTSAction`. 

At this point, you know enough about Dendron to start building applications of your own. We haven't talked about all of the different node types, but with the language model nodes we have introduced you can do quite a bit. In future tutorials we'll talk about "decorator nodes", Denron's support for GUI-based tree design, and Dendron's multimodal LM capabilities. If you can't wait for that last one, check out the API reference for `ImageLMAction` or the example notebook in the dendron repository. In the meantime, happy hacking!