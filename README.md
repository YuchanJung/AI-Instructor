# AI Instructor

> [!NOTE]
> This document is a report and code instruction of final Gemma project of Google ML Bootcamp 2024.

This project focuses on building an **AI Instructor, a Q&A bot**, using transcripts from the **Andrew Ng's Deep Learning course**. It was created specifically for the juniors of the Google ML Bootcamp to provide them with an interactive tool to deepen their understanding of key machine learning concepts. The provided model was created by fine-tuning the **Gemma-2B** model on a custom-generated Q&A dataset derived from the lecture content.

## Table of Contents

- [Project Overview]()
- [Data Collection]()
- [Data Preprocessing]()
- [Generate Q&A Dataset]()
- [Model Training]()
  - [Log]()
- [Reference]()



## Project Overview

1. **Lecture Transcript Collection**: We start by collecting the text data for lectures 1 through 10 from [Stanford CS230: Deep Learning series](https://www.youtube.com/playlist?list=PLoROMvodv4rOABXSygHTsbvUz4G_YQhOb) on YouTube. These lecture transcripts provide the foundation for generating our Q&A dataset.

2. **Data Preprocessing**: After extracting the raw transcript data, we perform several cleaning tasks such as removing time markers, unnecessary words, and formatting issues. The cleaned data is split into manageable chunks to facilitate the Q&A generation process.

3. **Q&A Dataset Generation**: Once the data is preprocessed, GPT-based assistants are employed to create Q&A pairs from the lecture content. This process involves feeding cleaned transcript chunks into the model and prompting it to generate questions and answers related to the lecture material.

4. **Model Training**: Finally, the generated Q&A pairs are used to train the Gemma 2B model, resulting in the creation of a fully functional AI Instructor capable of answering questions on AI and machine learning topics.


## Data Collection

The first step involves extracting the transcript data from YouTube. We crawled the transcripts for lectures 1-10 from the following YouTube playlist. The crawled transcripts are stored as .txt files for further processing.

## Data Preprocessing

Once the raw transcripts are gathered, we clean and process them to remove noise. This includes:
- Removing time markers (e.g., (10:35)).
- Stripping out filler words like "um" and "uh."
- Cleaning hyperlinks or irrelevant textual elements.
  
We also divide the cleaned transcripts into chunks, which are later used to feed into the GPT assistants for Q&A generation. Each lecture transcript is split into 5 chunks to ensure that the assistant receives manageable portions of data for processing.


## Generate Q&A Pairs with GPT Assistant

After cleaning and chunking the data, we leverage GPT-based assistants to generate a dataset of 30 Q&A pairs from the lecture content. The assistant receives the transcript chunks sequentially, acknowledges understanding, and then creates Q&A pairs focusing on core concepts in AI, relevant mathematics, and practical applications.

Here’s how the process works:

- Feed the assistant the first part of the transcript and wait for an acknowledgment.
- Continue feeding the rest of the chunks.
- Once all chunks are received, prompt the assistant to generate Q&A pairs in a JSON format.

**First Prompt**
```text
I will provide 5 consecutive lecture transcript parts. After each part, please respond with "Understood" if you have received and understood it. After I provide the fifth part, please generate a dataset of 30 Q&A pairs based on the entire transcript.
```

**Last Prompt**
```text
I have now provided all 5 parts of the transcript. Please create a 30 Q&A datasets focused on the principles of AI, relevant mathematics, and practical applications based on the lecture.

Make sure the questions cover both theoretical knowledge and how the concepts can be applied in real-world scenarios.

Format the response strictly in JSON, using the following structure:

[
    {
    "Question": "",
    "Answer": ""
    },
    {
    "Question": "",
    "Answer": ""
    },
    ...
]
```

The generated Q&A pairs are saved as a CSV file for easy access and further use in model training. Each row in the CSV contains a question and its corresponding answer.

## Model Training

Once the dataset is ready, we utilize it to fine-tune the Gemma 2B model. This allows the Q&A bot (AI Instructor) to respond to user questions about machine learning, based on the content from the lectures.