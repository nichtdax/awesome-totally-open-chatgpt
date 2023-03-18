# awesome-totally-open-chatgpt

ChatGPT is GPT-3.5 with RLHF (Reinforcement Learning with Human Feedback) for chat system.

By alternative, I mean projects feature different language model for chat system. 
Projects are **not** counted if they are:
- alternative frontend projects because they just call the API from OpenAI. 
- alternative transformer decoder models to GPT 3.5 either because the training data of them are (mostly) not for chat system.

Tags:

-   B: bare (no data, no model's weight, no chat system)
-   F: full (yes data, yes model's weight, yes chat system including TUI and GUI)

# The template

```markdown
## [{owner}/{project-name}]{https://github.com/link/to/project}

Lorem ipsum dolor sit amet.

Tags: B
```

# The list

## [lucidrains/PaLM-rlhf-pytorch](https://github.com/lucidrains/PaLM-rlhf-pytorch)

Implementation of RLHF (Reinforcement Learning with Human Feedback) on top of the PaLM architecture. Basically ChatGPT but with PaLM

Tags: B

## [togethercomputer/OpenChatKit](https://github.com/togethercomputer/OpenChatKit)

OpenChatKit provides a powerful, open-source base to create both specialized and general purpose chatbots for various applications. [Demo](https://huggingface.co/spaces/togethercomputer/OpenChatKit)

Tags: F

## [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui)

A gradio web UI for running Large Language Models like GPT-J 6B, OPT, GALACTICA, LLaMA, and Pygmalion.

Tags: F

## [KoboldAI/KoboldAI-Client](https://github.com/KoboldAI/KoboldAI-Client)

This is a browser-based front-end for AI-assisted writing with multiple local & remote AI models. It offers the standard array of tools, including Memory, Author’s Note, World Info, Save & Load, adjustable AI settings, formatting options, and the ability to import existing AI Dungeon adventures. You can also turn on Adventure mode and play the game like AI Dungeon Unleashed.

Tags: F

## [LAION-AI/Open-Assistant/](https://github.com/LAION-AI/Open-Assistant/) 

OpenAssistant is a chat-based assistant that understands tasks, can interact with third-party systems, and retrieve information dynamically to do so.

Tags: F
