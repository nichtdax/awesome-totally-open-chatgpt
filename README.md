<div align="center">
    <h1>Awesome Totally Open Chatgpt</h1>
    <a href="https://github.com/sindresorhus/awesome"><img src="https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg"/></a>
</div>

ChatGPT is GPT-3.5 finetuned with RLHF (Reinforcement Learning with Human Feedback) for human instruction and chat.

Alternatives are projects featuring different instruct finetuned language models for chat. 
Projects are **not** counted if they are:
- Alternative frontend projects which simply call OpenAI's APIs. 
- Using language models which are not finetuned for human instruction or chat.

Tags:
-   Bare: only, source, codeno data, no model's weight, no chat system
-   Standard: yes data, yes model's weight, bare chat via API
-   Full: full yes data, yes model's weight, fancy chat system including TUI and GUI
-   Complicated: semi open source, not really open source, based on closed model, etc...

# Table of Contents
1. [The template](#The-template)
2. [The list](#The-list)
   - [lucidrains/PaLM-rlhf-pytorch](#lucidrainsPaLM-rlhf-pytorch)
   - [togethercomputer/OpenChatKit](#togethercomputerOpenChatKit)
   - [oobabooga/text-generation-webui](#oobaboogatext-generation-webui)
   - [KoboldAI/KoboldAI-Client](#KoboldAIKoboldAI-Client)
   - [LAION-AI/Open-Assistant](#LAION-AIOpen-Assistant)
   - [tatsu-lab/stanford_alpaca](#tatsu-labstanford_alpaca)
   - [BlinkDL/ChatRWKV](#BlinkDLChatRWKV)
   - [THUDM/ChatGLM-6B](#THUDMChatGLM-6B)
   - [bigscience-workshop/xmtf](#bigscience-workshopxmtf)
   - [carperai/trlx](#carperaitrlx)

# The template

Append the new project at the end of file

```markdown
## [{owner}/{project-name}]{https://github.com/link/to/project}

Description goes here

Tags: Bare/Standard/Full/Complicated
```

# The list

## [lucidrains/PaLM-rlhf-pytorch](https://github.com/lucidrains/PaLM-rlhf-pytorch)

Implementation of RLHF (Reinforcement Learning with Human Feedback) on top of the PaLM architecture. Basically ChatGPT but with PaLM

Tags: Bare

## [togethercomputer/OpenChatKit](https://github.com/togethercomputer/OpenChatKit)

OpenChatKit provides a powerful, open-source base to create both specialized and general purpose chatbots for various applications. 

Related links:
- [spaces/togethercomputer/OpenChatKit](https://huggingface.co/spaces/togethercomputer/OpenChatKit)

Tags: Full

## [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui)

A gradio web UI for running Large Language Models like GPT-J 6B, OPT, GALACTICA, LLaMA, and Pygmalion.

Tags: Full

## [KoboldAI/KoboldAI-Client](https://github.com/KoboldAI/KoboldAI-Client)

This is a browser-based front-end for AI-assisted writing with multiple local & remote AI models. It offers the standard array of tools, including Memory, Author’s Note, World Info, Save & Load, adjustable AI settings, formatting options, and the ability to import existing AI Dungeon adventures. You can also turn on Adventure mode and play the game like AI Dungeon Unleashed.

Tags: Full

## [LAION-AI/Open-Assistant](https://github.com/LAION-AI/Open-Assistant) 

OpenAssistant is a chat-based assistant that understands tasks, can interact with third-party systems, and retrieve information dynamically to do so.

Related links:
- [huggingface.co/OpenAssistant](https://huggingface.co/OpenAssistant)
- [r/OpenAssistant/](https://www.reddit.com/r/OpenAssistant/)

Tags: Full

## [tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)

This is the repo for the Stanford Alpaca project, which aims to build and share an instruction-following LLaMA model. Related links:
- [pointnetwork/point-alpaca](https://github.com/pointnetwork/point-alpaca)
- [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora)
- [r/LocalLLaMA How to install LLaMA: 8-bit and 4-bit](https://www.reddit.com/r/LocalLLaMA/comments/11o6o3f/how_to_install_llama_8bit_and_4bit/)
- [antimatter15/alpaca.cpp](https://github.com/antimatter15/alpaca.cpp)
- [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
- [setzer22/llama-rs](https://github.com/setzer22/llama-rs)
- [juncongmoo/chatllama](https://github.com/juncongmoo/chatllama)

See these Reddit comments first [#1](https://www.reddit.com/r/MachineLearning/comments/11uk8ti/comment/jcpd3yu/?utm_source=share&utm_medium=web2x&context=3)

Tags: Complicated

## [BlinkDL/ChatRWKV](https://github.com/BlinkDL/ChatRWKV)

ChatRWKV is like ChatGPT but powered by RWKV (100% RNN) language model, and open source.

Tags: Full

## [THUDM/ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)

ChatGLM-6B is an open bilingual language model based on General Language Model (GLM) framework, with 6.2 billion parameters. With the quantization technique, users can deploy locally on consumer-grade graphics cards (only 6GB of GPU memory is required at the INT4 quantization level).

Related links:

- Alternative Web UI: [Akegarasu/ChatGLM-webui](https://github.com/Akegarasu/ChatGLM-webui)
- Slim version (remove 20K image tokens to reduce memory usage): [silver/chatglm-6b-slim](https://huggingface.co/silver/chatglm-6b-slim)
- Fintune ChatGLM-6b using low-rank adaptation (LoRA): [lich99/ChatGLM-finetune-LoRA](https://github.com/lich99/ChatGLM-finetune-LoRA)
- Deploying ChatGLM on Modelz: [tensorchord/modelz-ChatGLM](https://github.com/tensorchord/modelz-ChatGLM)
- Docker image with built-on playground UI and streaming API compatible with OpenAI, using [Basaran](https://github.com/hyperonym/basaran): [peakji92/chatglm:6b](https://hub.docker.com/r/peakji92/chatglm/tags)

Tags: Full

## [bigscience-workshop/xmtf](https://github.com/bigscience-workshop/xmtf)

This repository provides an overview of all components used for the creation of BLOOMZ & mT0 and xP3 introduced in the paper [Crosslingual Generalization through Multitask Finetuning](https://arxiv.org/abs/2211.01786).

Related links:
- [bigscience/bloomz](https://huggingface.co/bigscience/bloomz)
- [bigscience/mt0-base](https://huggingface.co/bigscience/mt0-base)

Tags: Standard

## [carperai/trlx](https://github.com/carperai/trlx)

 A repo for distributed training of language models with Reinforcement Learning via Human Feedback (RLHF), supporting online RL up to 20b params and offline RL to larger models. Basically what you would use to finetune GPT into ChatGPT. 

 Tags: Bare
