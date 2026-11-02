# LLM Course: Assignments and Projects

Coursework for the Large Language Models course (Spring 2024). Each notebook implements one assignment end to end, from data preparation to training and evaluation, with short written answers to the conceptual questions that accompany it. Notebooks are written for Google Colab with a GPU.

## Contents

| Notebook | Topic | What is implemented |
|---|---|---|
| `LLM_CA1.ipynb` | Embeddings and transfer learning | GloVe nearest neighbours and analogies; masked-LM probing; fine-tuning BERT for text classification and for extractive question answering (SQuAD-style) |
| `LLM_CA2_gpt.ipynb` | Efficient generation | Single-sentence and batched generation with GPT-2, with and without the `past_key_values` cache, and a tokens-per-second comparison |
| `LLM_CA2_Soft_Prompt.ipynb` | Soft prompts | A trainable prompt-embedding layer written from scratch and swapped into a frozen Persian BERT (`bert-fa-base-uncased`) for text classification; the same method re-run with the OpenDelta library (data in `softprompt_dataset.csv`) |
| `LLM_CA3.ipynb` | Reasoning, PEFT, RAG | Chain-of-thought prompting; LoRA fine-tuning of Phi-2 with `peft` and `trl`; a retrieval-augmented generation pipeline with LangChain and a local quantised Llama-2-7B-Chat (GGUF), comparing TF-IDF and semantic (`e5-base`) retrievers on mini-BioASQ, plus the effect of temperature and out-of-domain questions |
| `LLM_CA4.ipynb` | RLHF, quantisation, evaluation | The full RLHF pipeline for summarisation on the TL;DR data: supervised fine-tuning of GPT-2, a pairwise reward model, and PPO; 4-bit (NF4, double quantisation) QLoRA fine-tuning of Mistral-7B; instruction tuning; evaluating generated text with BERTScore and a language model as judge |

## Main techniques

Parameter-efficient fine-tuning (soft prompts, LoRA, QLoRA), RLHF (SFT, reward modelling, PPO), 4-bit quantisation, KV-cache generation, retrieval-augmented generation, and automatic evaluation of generated text.

