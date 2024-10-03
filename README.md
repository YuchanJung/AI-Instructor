# AI Instructor

> [!NOTE]
> This document is a report and code instruction of final Gemma project of Google ML Bootcamp 2024.

This project focuses on building an **AI Instructor, a Q&A bot**, using transcripts from the **Andrew Ng's Deep Learning course**. It was created specifically for the juniors of the Google ML Bootcamp to provide them with an interactive tool to deepen their understanding of key machine learning concepts. The provided model was created by fine-tuning the **Gemma-2B** model on a custom-generated Q&A dataset derived from the lecture content.

# Table of Contents

- [AI Instructor](#ai-instructor)
- [Table of Contents](#table-of-contents)
- [Project Overview](#project-overview)
- [Detailed Workflow](#detailed-workflow)
  - [Data Collection](#data-collection)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training](#model-training)
  - [Results](#results)
  - [Model Performance Before and After Fine-Tuning](#model-performance-before-and-after-fine-tuning)



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


### Generate Q&A Pairs with GPT Assistant

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
| <img width="801" alt="image1" src="https://github.com/user-attachments/assets/9fb5b267-71e7-4505-9a45-883401e16435"> | `{"Question: What is a logistic regression model?\nAnswer: It’s a basic machine learning model for classification.", "Question: What does CNN stand for?\nAnswer: Convolutional Neural Network.", "Question: What is a loss function?\nAnswer: A function that measures model performance by comparing predictions to true values."}` |


### Tokenization
Tokenization is the process of converting raw text into tokens that the model can understand. In this case, the AutoTokenizer from Hugging Face is used to handle the tokenization process. The tokenizer breaks down text (questions and answers) into smaller units called tokens, which can be either words, subwords, or even characters, depending on the model and tokenizer configuration.

The model used here is a causal language model, which typically uses a byte-pair encoding (BPE) tokenizer. This means that common words may remain whole, while rare words are split into subword units.

| Before (text) | After (token) | 
| -- | -- |
| `Question: What is a logistic regression model?\nAnswer: It’s a basic machine learning model for classification.` | `{'input_ids': [0, 2, 9413, 235292, 2439, 603, 573, 6045, 576, 15155, 235284, 235304, 235276, 235336, 108, 1261, 235292, 15155, 235284, 235304, 235276, 31381, 611, 5271, 6044, 578, 11572, 8557, 235265], 'attention_mask': [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }` |



## Model Training

Once the dataset is ready, we utilize it to fine-tune the `Gemma-2B model`. This allows the Q&A bot (AI Instructor) to respond to user questions about machine learning, based on the content from the lectures.

### Training Process Overview

The fine-tuning process was performed on the `Gemma-2B model` using Hugging Face's Trainer API. The model was trained for `3 epochs`, with the batch size set to 1 for a more focused training on individual Q&A pairs. The training arguments included:
- **Output Directory**: All model checkpoints and outputs were saved in the specified directory.
- **Learning Rate**: Set to 5e-5, a common rate for fine-tuning large models.
- **Epochs**: Trained over 3 epochs to ensure convergence without overfitting.
- **Logging**: To monitor the training process, loss values were logged every 100 steps.
- **Saving Checkpoints**: The model's state was saved every 500 steps, keeping a maximum of two checkpoints to manage memory efficiently.

### Code
```python
def fine_tune_model(model, dataset, tokenizer, output_dir="./results", epochs=3):
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        learning_rate=5e-5,
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )
    
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

fine_tune_model(model, dataset, tokenizer, output_dir, epochs=3)
```

## Results

The training loss decreased steadily over time, as visualized in the provided graph:

The final loss settled around 0.5, showing effective learning from the Q&A dataset generated from the course transcripts.
### Optimization Log

![training_loss](https://github.com/user-attachments/assets/072ac5eb-d953-4797-a5c0-691374e812b4)

## Model Performance Before and After Fine-Tuning

Here’s a comparison of the model’s performance on two sample questions before and after fine-tuning:

### 1. Question: ***Why do we use CNN?***
Before Fine-Tuning:
```text
Why are Convolutional Neural Networks so effective? A neural network is a collection of simple processing units. It is an approximation of biological neural networks.

In biological systems, we can see the network of neurons. Each neuron is connected to other neurons through a number of axons. These neurons are responsible for processing information and making decisions. For example, the brain of a mouse is composed of 100 billion neurons, each neuron has about 50 ...
```
After Fine-tuning
```text
CNNs are used to detect visual information from raw data. They can detect local patterns in data, especially images. In the provided example, you can see the features that maximally activate neurons. The maximum to minimum curvature. What is a loss function for the neural network. It measures the difference between predicted and true values. A function measuring the model's performance. To detect the maximum, minimum, and average values in the input
```


### 2. Question: ***What is Reinforcement Learning?***
Before Fine-Tuning:
```text
It keeps networks stronger, better interpret the model, and participate in communities. Follow communities like subreddits, like like model class, participate, models discuss Research papers, stay informed, enough, more, in the chat, or stay current on the latest conference proceedings...
```
After Fine-tuning:
```text
Reinforcement learning is a machine learning method used to predict the behavior of an agent in a complex, dynamic environment based on its past experiences. Reinforcement Learning aims to maximize the reward of the agent over the long term. A reward is defined as a positive or negative value that is given to the system for taking a certain action in the environment. The reward value of a given action is used as feedback to update the actions of that agent.
```





