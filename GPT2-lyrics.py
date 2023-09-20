# %%
#!pip install transformers[torch]
#!pip install accelerate==0.20.1
#!pip install datasets

# %%
import torch, os, re, pandas as pd, json
from sklearn.model_selection import train_test_split
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding, GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

import accelerate
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# %%
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


# %% [markdown]
# ### Taylor Swift

# %%
taylor = pd.read_csv('taylor_swift_lyrics.csv', encoding = "latin1")
taylor.head(10)

# %%
aux = []
phrase = "<Taylor Swift>"
for i in range(0,taylor["lyric"].size):
  phrase = phrase + taylor["lyric"][i] + " "
  if len(phrase) > 150:
    aux.append(phrase)
    phrase = "<{}>".format(taylor["album"][i])
data = pd.DataFrame(aux, columns=["data"])
data["data"][452]

# %% [markdown]
# ### GPT2

# %%
# Download pre-trained model
MODEL = 'gpt2'
# 'distilgpt2' 'gpt2' 'gpt2-medium' 'gpt2-large' 'gpt2-xl'
# 'microsoft/DialoGPT-small' 'microsoft/DialoGPT-medium' 'microsoft/DialoGPT-large'


model = AutoModelForCausalLM.from_pretrained(MODEL)

# %% [markdown]
# # Tokenizar
# 
# 
# 

# %%
# We load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# %%
taylor["album"].unique()

# %%
bos = '<bos>'
eos = '<eos>'
pad = '<pad>'

# Tokenizer preparation
special_tokens_dict = {'additional_special_tokens': ['<pad>', '<bos>', '<eos>', '<sep>', '<Taylor Swift>', '<Fearless>', '<Speak Now>', '<Red>', '<1989>', '<reputation>']}
tokenizer.add_special_tokens(special_tokens_dict)
tokenizer.additional_special_tokens = special_tokens_dict['additional_special_tokens']

tokenizer.pad_token = '<pad>'
tokenizer.bos_token = "<bos>"
tokenizer.eos_token = "<eos>"

model.resize_token_embeddings(len(tokenizer))


# %%
#Add the tokes of starting a sentence and end of a sentence to the data
data['data'] = bos + ' ' + data['data'] + ' ' + eos

df_train, df_val = train_test_split(data, train_size = 0.9, random_state = 77)
print(f'There are {len(df_train)} sentences for training and {len(df_val)} for validation')


# %%
# Create dataset for the training
train_dataset = Dataset.from_pandas(df_train[['data']])
val_dataset = Dataset.from_pandas(df_val[['data']])


# %%
#Tokenize
def tokenize_function(examples):
        return tokenizer(examples['data'], padding=True, max_length=200 , truncation=True)


tokenized_train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    num_proc=5,
    remove_columns=['data'],
)
tokenized_test_dataset = val_dataset.map(
    tokenize_function,
    batched=True,
    num_proc=5,
    remove_columns=['data'],
)

# %%

def add_labels(dataset):
  result = {'attention_mask':dataset['attention_mask'], 'input_ids':dataset['input_ids']}
  result['labels']=dataset['input_ids'].copy()
  return result

train = tokenized_train_dataset.map(add_labels , batched=True)
test = tokenized_test_dataset.map(add_labels , batched=True)

# %%
# Example of the result of the tokenization process with padding
tokenizer.decode(tokenized_train_dataset['input_ids'][34])


# %% [markdown]
# # Train

# %%
# https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments

model_path = './model_lyrics_TS'

training_args = TrainingArguments(
    output_dir=model_path,          # output directory
    num_train_epochs=6,              # total # of training epochs
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=200,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir=model_path,            # directory for storing logs
    #prediction_loss_only=True,
    save_steps=10000,
    save_total_limit = 2,	      # total checkpoints saved, deletes older ones
    learning_rate = 2e-4              # default: 5e-5
)


# %%
data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )


# %%
#Metrics of quality
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# %%
trainer = Trainer(
    model=model,                         # the instantiated  Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    data_collator=data_collator,
    #compute_metrics=compute_metrics,
    train_dataset=tokenized_train_dataset,         # training dataset
    eval_dataset=tokenized_test_dataset            # evaluation dataset
)
trainer.train()

# %%
#Save the model and tokenizer
trainer.save_model()
tokenizer.save_pretrained(model_path)


# %%
evaluation = trainer.evaluate()
print(evaluation)

# %%
#Get the evaluation loss and perplexity

eval_loss = evaluation['eval_loss']
perplexity = torch.exp(torch.tensor(eval_loss))
print(f"Evaluation loss: {eval_loss}, Perplexity: {perplexity}")

# %% [markdown]
# # Generation

# %%
#Function for generation of lyrics

def generate_n_text_samples(model, tokenizer, input_text, device, n_samples = 5):
    text_ids = tokenizer.encode(input_text, return_tensors = 'pt')
    text_ids = text_ids.to(device)
    model = model.to(device)

    generated_text_samples = model.generate(
        text_ids,
        max_length= 50,
        num_return_sequences= n_samples,
        no_repeat_ngram_size= 2,
        repetition_penalty= 1.5,
        top_p= 0.92,
        temperature= .85,
        do_sample= True,
        top_k= 125,
        early_stopping= True
    )
    gen_text = []
    for t in generated_text_samples:
        text = tokenizer.decode(t, skip_special_tokens=True)
        gen_text.append(text)

    return gen_text

# %%
# trained model loading
model_path = './model_lyrics_TS'


model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

device = "cuda:0"

input_text = tokenizer.bos_token + "<Red>"

# %%
sentences = generate_n_text_samples(model, tokenizer,
                                    input_text, device, n_samples = 5)
for h in sentences:
    print(h)
    print()


