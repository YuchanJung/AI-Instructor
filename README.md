# AI Instructor

> [!NOTE]
> This document is a report and code instruction of final Gemma project of Google ML Bootcamp 2024.

This project focuses on building an **AI Instructor, a Q&A bot**, using transcripts from the **Andrew Ng's Deep Learning course**. It was created specifically for the juniors of the Google ML Bootcamp to provide them with an interactive tool to deepen their understanding of key machine learning concepts. The provided model was created by fine-tuning the **Gemma-2B** model on a custom-generated Q&A dataset derived from the lecture content.

# Table of Contents

- [Project Overview](#Project-Overview)
- [Detailed Workflow](#Detailed-Workflow)
  - [Data Collection](#Data-Collection)
  - [Data Preprocessing](#Data-Preprocessing)
  - [Generate Q&A Dataset](#Generate-Q&A-Pairs-with-GPT-Assistant)
  - [Model Training](#Model-Training)
    - [Log]()
  - [Reference]()



# Project Overview

1. **Lecture Transcript Collection**: We start by collecting the text data for lectures 1 through 10 from [Stanford CS230: Deep Learning series](https://www.youtube.com/playlist?list=PLoROMvodv4rOABXSygHTsbvUz4G_YQhOb) on YouTube. These lecture transcripts provide the foundation for generating our Q&A dataset.

2. **Data Preprocessing**: After extracting the raw transcript data, we perform several cleaning tasks such as removing time markers, unnecessary words, and formatting issues. The cleaned data is split into manageable chunks to facilitate the Q&A generation process.

3. **Q&A Dataset Generation**: Once the data is preprocessed, GPT-based assistants are employed to create Q&A pairs from the lecture content. This process involves feeding cleaned transcript chunks into the model and prompting it to generate questions and answers related to the lecture material.

4. **Model Training**: Finally, the generated Q&A pairs are used to train the Gemma 2B model, resulting in the creation of a fully functional AI Instructor capable of answering questions on AI and machine learning topics.

# Detailed Workflow
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



## Pre-processing

### Load and Setup the model
The model and tokenizer used for fine-tuning are loaded from the Hugging Face model hub. In this example, the model is google/gemma-2-2b, a large causal language model. 


### Transforming Q&A pair to LLM input
```python
def load_qna_files(data_dir):
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    data = []
    for file in files:
        dataset = pd.read_csv(file)
        print(f'Sample data of {file}')
        print(dataset.head(5))
        for index, row in dataset.iterrows():
            data.append(f"Question: {row['Question']}\nAnswer: {row['Answer']}")
    return data
```
 Each file contains a Q&A format, and the data is processed into the appropriate format for language modeling. Specifically, each row in the dataset is converted into a string with the format: `"Question: [Question]\nAnswer: [Answer]".`

**Example**: 

| Before (csv) | After (text) |
| -- | -- |
| ![image1](https://github.com/user-attachments/assets/46d1e3e4-88dc-455d-8d6d-58144146fa71") | `{"Question: What is a logistic regression model?\nAnswer: It’s a basic machine learning model for classification.", "Question: What does CNN stand for?\nAnswer: Convolutional Neural Network.", "Question: What is a loss function?\nAnswer: A function that measures model performance by comparing predictions to true values."}` |


### Tokenization
Tokenization is the process of converting raw text into tokens that the model can understand. In this case, the AutoTokenizer from Hugging Face is used to handle the tokenization process. The tokenizer breaks down text (questions and answers) into smaller units called tokens, which can be either words, subwords, or even characters, depending on the model and tokenizer configuration.

The model used here is a causal language model, which typically uses a byte-pair encoding (BPE) tokenizer. This means that common words may remain whole, while rare words are split into subword units.

| Before (text) | After (token) | 
| -- | -- |
| `Question: What is a logistic regression model?\nAnswer: It’s a basic machine learning model for classification.` | `{'input_ids': [0, 2, 9413, 235292, 2439, 603, 573, 6045, 576, 15155, 235284, 235304, 235276, 235336, 108, 1261, 235292, 15155, 235284, 235304, 235276, 31381, 611, 5271, 6044, 578, 11572, 8557, 235265], 'attention_mask': [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }` |

