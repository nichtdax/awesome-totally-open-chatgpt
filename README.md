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
-   Bare: only source code, no data, no model's weight, no chat system
-   Standard: yes data, yes model's weight, bare chat via API
-   Full: full yes data, yes model's weight, fancy chat system including TUI and GUI
-   Complicated: semi open source, not really open source, based on closed model, etc...

Other revelant lists:
- [yaodongC/awesome-instruction-dataset](https://github.com/yaodongC/awesome-instruction-dataset): A collection of open-source dataset to train instruction-following LLMs (ChatGPT,LLaMA,Alpaca)

# Table of Contents
1. [The template](#The-template)
2. [The list](#The-list)
   - [lucidrains/PaLM-rlhf-pytorch](#lucidrainsPaLM-rlhf-pytorch)
   - [togethercomputer/OpenChatKit](#togethercomputerOpenChatKit)
   - [oobabooga/text-generation-webui](#oobaboogatext-generation-webui)
   - [KoboldAI/KoboldAI-Client](#KoboldAIKoboldAI-Client)
   - [LAION-AI/Open-Assistant](#LAION-AIOpen-Assistant)
   - [tatsu-lab/stanford_alpaca](#tatsu-labstanford_alpaca)
     - [Other LLaMA-derived projects](#other-llama-derived-projects)
   - [BlinkDL/ChatRWKV](#BlinkDLChatRWKV)
   - [THUDM/ChatGLM-6B](#THUDMChatGLM-6B)
   - [bigscience-workshop/xmtf](#bigscience-workshopxmtf)
   - [carperai/trlx](#carperaitrlx)
   - [databrickslabs/dolly](#databrickslabsdolly)
   - [LianjiaTech/BELLE](#lianjiatechbelle)
   - [ethanyanjiali/minChatGPT](#ethanyanjialiminchatgpt)
   - [cerebras/Cerebras-GPT](#cerebrascerebras-gpt)
   - [TavernAI/TavernAI](#tavernaitavernai)
   - [Cohee1207/SillyTavern](#cohee1207sillytavern)
   - [h2oai/h2ogpt](#h2oaih2ogpt)
   - [mlc-ai/web-llm](#mlc-aiweb-llm)
   - [Stability-AI/StableLM](#stability-aistablelm)
   - [clue-ai/ChatYuan](#clue-aichatyuan)
   - [OpenLMLab/MOSS](#openlmlabmoss)

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

This is the repo for the Stanford Alpaca project, which aims to build and share an instruction-following LLaMA model.

Tags: Complicated

### Other LLaMA-derived projects:

- [pointnetwork/point-alpaca](https://github.com/pointnetwork/point-alpaca) Released weights recreated from Stanford Alpaca, an experiment in fine-tuning LLaMA on a synthetic instruction dataset.
- [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora) Code for rproducing the Stanford Alpaca results using low-rank adaptation (LoRA).
- [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) Ports for inferencing LLaMA in C/C++ running on CPUs, supports alpaca, gpt4all, etc.
- [setzer22/llama-rs](https://github.com/setzer22/llama-rs) Rust port of the llama.cpp project.
- [juncongmoo/chatllama](https://github.com/juncongmoo/chatllama) Open source implementation for LLaMA-based ChatGPT runnable in a single GPU.
- [Lightning-AI/lit-llama](https://github.com/Lightning-AI/lit-llama) Implementation of the LLaMA language model based on nanoGPT.
- [nomic-ai/gpt4all](https://github.com/nomic-ai/gpt4all) Demo, data and code to train an assistant-style large language model with ~800k GPT-3.5-Turbo Generations based on LLaMA.
- [hpcaitech/ColossalAI#ColossalChat](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat) An open-source solution for cloning ChatGPT with a complete RLHF pipeline.
- [lm-sys/FastChat](https://github.com/lm-sys/FastChat) An open platform for training, serving, and evaluating large language model based chatbots.
- [nsarrazin/serge](https://github.com/nsarrazin/serge) A web interface for chatting with Alpaca through llama.cpp. Fully dockerized, with an easy to use API.

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

## [databrickslabs/dolly](https://github.com/databrickslabs/dolly)

Databricks’ dolly-v2-12b, an instruction-following large language model trained on the Databricks machine learning platform that is licensed for commercial use. Based on pythia-12b trained on ~15k instruction/response fine tuning records [databricks-dolly-15k](https://github.com/databrickslabs/dolly/tree/master/data) generated by Databricks employees in capability domains from the InstructGPT paper.

Related links:
- [dolly v2 12B commercial commercially available model](https://huggingface.co/databricks/dolly-v2-12b)
- [dolly v1 6b model card](https://huggingface.co/databricks/dolly-v1-6b)

Tags: Standard

## [LianjiaTech/BELLE](https://github.com/LianjiaTech/BELLE)

The goal of this project is to promote the development of the open-source community for Chinese language large-scale conversational models. This project optimizes Chinese performance in addition to original Stanford Alpaca. The model finetuning uses only data generated via ChatGPT (without other data). This repo contains: 175 chinese seed tasks used for generating the data, code for generating the data, 0.5M generated data used for fine-tuning the model, model finetuned from BLOOMZ-7B1-mt on data generated by this project.

Related links:
- [English readme](https://github.com/LianjiaTech/BELLE#-belle-be-large-language-model-engine-1)

Tags: Standard

## [ethanyanjiali/minChatGPT](https://github.com/ethanyanjiali/minChatGPT)

A minimum example of aligning language models with RLHF similar to ChatGPT

Related links:
- [huggingface.co/ethanyanjiali/minChatGPT](https://huggingface.co/ethanyanjiali/minChatGPT)

Tags: Standard

## [cerebras/Cerebras-GPT](https://huggingface.co/cerebras/Cerebras-GPT-6.7B)

7 open source GPT-3 style models with parameter ranges from 111 million to 13 billion, trained using the [Chinchilla](https://arxiv.org/abs/2203.15556) formula. Model weights have been released under a permissive license (Apache 2.0 license in particular).

Related links:
- [Announcement](https://www.cerebras.net/blog/cerebras-gpt-a-family-of-open-compute-efficient-large-language-models/)
- [Models with other amount of parameters](https://huggingface.co/cerebras)

Tags: Standard

## [TavernAI/TavernAI](https://github.com/TavernAI/TavernAI)

Atmospheric adventure chat for AI language model **Pygmalion** by default and other models such as **KoboldAI**, ChatGPT, GPT-4

Tags: Full

## [Cohee1207/SillyTavern](https://github.com/Cohee1207/SillyTavern)

SillyTavern is a fork of TavernAI 1.2.8 which is under more active development, and has added many major features. At this point they can be thought of as completely independent programs. On its own Tavern is useless, as it's just a user interface. You have to have access to an AI system backend that can act as the roleplay character. There are various supported backends: OpenAPI API (GPT), KoboldAI (either running locally or on Google Colab), and more.

Tags: Full

## [h2oai/h2ogpt](https://github.com/h2oai/h2ogpt)

h2oGPT - The world's best open source GPT
- Open-source repository with fully permissive, commercially usable code, data and models
- Code for preparing large open-source datasets as instruction datasets for fine-tuning of large language models (LLMs), including prompt engineering
- Code for fine-tuning large language models (currently up to 20B parameters) on commodity hardware and enterprise GPU servers (single or multi node)
- Code to run a chatbot on a GPU server, with shareable end-point with Python client API
- Code to evaluate and compare the performance of fine-tuned LLMs

Related links:
- [h2oGPT 20B](https://gpt.h2o.ai/)
- [🤗 h2oGPT 12B #1](https://huggingface.co/spaces/h2oai/h2ogpt-chatbot)
- [🤗 h2oGPT 12B #2](https://huggingface.co/spaces/h2oai/h2ogpt-chatbot2)

Tags: Full

## [mlc-ai/web-llm](https://github.com/mlc-ai/web-llm)

Bringing large-language models and chat to web browsers. Everything runs inside the browser with no server support.

Related links:
- https://mlc.ai/web-llm

Tags: Full

## [Stability-AI/StableLM](https://github.com/Stability-AI/StableLM)

This repository contains Stability AI's ongoing development of the StableLM series of language models and will be continuously updated with new checkpoints.

Related links:
- [huggingface.co/spaces/stabilityai/stablelm-tuned-alpha-chat](https://huggingface.co/spaces/stabilityai/stablelm-tuned-alpha-chat)

Tags: Full

## [clue-ai/ChatYuan](https://github.com/clue-ai/ChatYuan)

ChatYuan: Large Language Model for Dialogue in Chinese and English (The repos are mostly in Chinese)

Related links:
- [A bit translated readme to English](https://github.com/nichtdax/awesome-totally-open-chatgpt/issues/18#issuecomment-1492826662)

Tags: Full


## [OpenLMLab/MOSS](https://github.com/OpenLMLab/MOSS)

MOSS: An open-source tool-augmented conversational language model from Fudan University. (Most examples are in Chinese)

Related links:
- [English readme](https://github.com/OpenLMLab/MOSS/blob/main/README_en.md)

Tags: Full

