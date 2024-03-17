# Script to evaluate the model using the test set.
# Load the model, make predictions, and calculate performance metrics.

# evaluate.py
import torch
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3TokenizerFast
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from seqeval.metrics import classification_report
from data_preparation import load_data  # Make sure this matches your project's structure
from config import MODEL_NAME, TRAINING_OUTPUT_DIR
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
import numpy as np

def prepare_dataset(tokenizer, texts, labels):
    """
    Prepares the dataset for evaluation, similar to the training preparation.
    """
    # Your implementation here, should be similar to the one in train.py
    pass  # Placeholder

def evaluate(model, dataset, tokenizer, label_list):
    """
    Evaluates the model performance on the provided dataset.
    """
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=8)

    model.eval()

    predictions , true_labels = [], []

    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)
        
        logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        # Collect predictions and true labels for each batch
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend(label_ids)
    
    # Convert indices to labels
    pred_labels = [[label_list[p_i] for p_i, _ in enumerate(batch)] for batch in predictions]
    true_labels = [[label_list[l_i] for l_i, _ in enumerate(batch)] for batch in true_labels]

    # Use seqeval's classification report for sequence labeling
    print("Evaluation metrics:")
    print(classification_report(true_labels, pred_labels))

def main():
    # Load your validation dataset
    val_texts, val_labels = load_data(split='validation')  # Update according to your data loading method

    # Load the trained model and tokenizer
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(TRAINING_OUTPUT_DIR)
    model = LayoutLMv3ForTokenClassification.from_pretrained(TRAINING_OUTPUT_DIR)
    
    # Assuming you've saved your label list during training
    label_list = ['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']  # Example labels

    # Prepare the validation dataset
    val_dataset = prepare_dataset(tokenizer, val_texts, val_labels)

    # Evaluate the model
    evaluate(model, val_dataset, tokenizer, label_list)

if __name__ == "__main__":
    main()
