# Wyckoff_chatbot/train_model.py

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils.model_handler import Transformer, CustomTokenizer

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters (reduced size for faster training)
MAX_LEN = 40
NUM_HEADS = 4  # Reduced from 8
D_MODEL = 256  # Reduced from 512
FFN_UNITS = 512  # Reduced from 2048
DROPOUT = 0.1
NUM_LAYERS = 3  # Reduced from 6
VOCAB_SIZE = 8000
BATCH_SIZE = 16
EPOCHS = 5  # Reduced from 10
LEARNING_RATE = 0.0001

def prepare_data():
    # Load the dataset
    data = pd.read_csv('assets/Cleaned_Wyckoff_QA_Dataset.csv')
    questions = data['Question'].astype(str).tolist()
    answers = data['Answer'].astype(str).tolist()
    
    # Create and fit tokenizer
    tokenizer = CustomTokenizer(VOCAB_SIZE)
    tokenizer.fit_on_texts(questions + answers)
    
    # Convert text to sequences
    question_seqs = tokenizer.texts_to_sequences(questions)
    answer_seqs = tokenizer.texts_to_sequences(answers)
    
    # Pad sequences
    def pad_sequences(sequences, maxlen):
        padded = np.zeros((len(sequences), maxlen))
        for i, seq in enumerate(sequences):
            if len(seq) > maxlen:
                padded[i] = seq[:maxlen]
            else:
                padded[i, :len(seq)] = seq
        return padded
    
    question_data = pad_sequences(question_seqs, MAX_LEN)
    answer_data = pad_sequences(answer_seqs, MAX_LEN)
    
    return torch.tensor(question_data, dtype=torch.long), torch.tensor(answer_data, dtype=torch.long), tokenizer

def train():
    # Prepare data
    print("Preparing data...")
    encoder_input, decoder_input, tokenizer = prepare_data()
    
    # Create model
    print("Creating model...")
    model = Transformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        ffn_units=FFN_UNITS,
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT,
        max_len=MAX_LEN
    ).to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("Starting training...")
    n_batches = len(encoder_input) // BATCH_SIZE
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for i in tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{EPOCHS}"):
            start_idx = i * BATCH_SIZE
            end_idx = start_idx + BATCH_SIZE
            
            batch_encoder_input = encoder_input[start_idx:end_idx].to(device)
            batch_decoder_input = decoder_input[start_idx:end_idx].to(device)
            
            # Forward pass
            output = model(batch_encoder_input, batch_decoder_input)
            
            # Calculate loss
            loss = criterion(output.view(-1, VOCAB_SIZE), batch_decoder_input.view(-1))
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch+1}/{EPOCHS}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f'assets/wyckoff_model_checkpoint_{epoch+1}.pth')
    
    # Save the final model
    print("Saving final model...")
    torch.save(model.state_dict(), 'assets/wyckoff_model.pth')
    print("Training completed!")

if __name__ == "__main__":
    train()