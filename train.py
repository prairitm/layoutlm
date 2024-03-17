# from sklearn.model_selection import train_test_split
# import pandas as pd
# import numpy as np
# from utils import calculate_iou
# from data_preparation import custom_dataset, label2id
# import json
# import os

# Code to train the model.
# This includes splitting the dataset, training the model, and saving the model weights.

# train.py
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3TokenizerFast, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
import os
from config import MODEL_NAME, TRAINING_OUTPUT_DIR
from data_preparation import load_data  # Make sure to implement this based on your project specifics

def prepare_dataset(tokenizer, texts, labels):
    """
    Prepare the dataset for training.

    Args:
    - tokenizer: The tokenizer for encoding the texts.
    - texts: A list of texts to be tokenized and encoded.
    - labels: The labels for the input texts.

    Returns:
    - TensorDataset: The TensorDataset containing input_ids, attention_masks, and labels.
    """
    input_ids = []
    attention_masks = []
    label_masks = []

    for i, text in enumerate(texts):
        encoded_dict = tokenizer.encode_plus(
            text,                      # Input text
            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
            max_length = 128,          # Pad & truncate all sentences.
            pad_to_max_length = True,
            return_attention_mask = True,   # Construct attn. masks.
            return_tensors = 'pt',     # Return pytorch tensors.
        )
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
        
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])

        # Create a mask for the labels (0 for padding, 1 for real label)
        label_mask = [float(i != tokenizer.pad_token_id) for i in encoded_dict['input_ids']]
        label_masks.append(label_mask)

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    label_masks = torch.tensor(label_masks)

    # Create the DataLoader.
    dataset = TensorDataset(input_ids, attention_masks, labels, label_masks)
    return dataset

def train(model, dataset, tokenizer, batch_size=16, epochs=4, lr=5e-5):
    """
    Train the model on the dataset.

    Args:
    - model: The model to train.
    - dataset: The training dataset.
    - tokenizer: The tokenizer for encoding the texts.
    - batch_size: The batch size for training.
    - epochs: The number of epochs to train for.
    - lr: The learning rate for the Adam optimizer.
    """
    train_dataloader = DataLoader(
            dataset,  # The training samples.
            sampler = RandomSampler(dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

    # Total number of training steps is [number of batches] x [number of epochs].
    total_steps = len(train_dataloader) * epochs

    # Create the optimizer
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)

    # Training loop
    model.train()
    for epoch_i in range(0, epochs):
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        total_loss = 0

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            b_label_mask = batch[3].to(device)

            model.zero_grad()        

            # Perform a forward pass (evaluate the model on this training batch).
            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=b_labels)

            loss = outputs[0]
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)            
        print("Average train loss: {}".format(avg_train_loss))

    print("Training complete!")

def main():
    # Load your dataset
    texts, labels = load_data()  # Implement this function based on your dataset

    # Split dataset into training and validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1)

    # Load the pre-trained model and tokenizer
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(MODEL_NAME)
    model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_NAME)

    # Prepare datasets
    train_dataset = prepare_dataset(tokenizer, train_texts, train_labels)
    val_dataset = prepare_dataset(tokenizer, val_texts, val_labels)

    # Train the model
    train(model, train_dataset, tokenizer)

    # Save the trained model and the tokenizer
    model.save_pretrained(TRAINING_OUTPUT_DIR)
    tokenizer.save_pretrained(TRAINING_OUTPUT_DIR)

if __name__ == '__main__':
    main()
