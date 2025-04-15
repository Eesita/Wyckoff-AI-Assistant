import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
import pandas as pd
import re

logger = logging.getLogger(__name__)

# Define all model components from your notebook
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :].to(x.device)

def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))
    dk = q.size(-1)
    scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(dk, dtype=torch.float32, device=q.device))
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask):
        batch_size = q.size(0)
        q = self.wq(q).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        k = self.wk(k).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        v = self.wv(v).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        attention = scaled_dot_product_attention(q, k, v, mask)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.dense(attention)

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, ffn_units, dropout_rate):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, ffn_units)
        self.linear2 = nn.Linear(ffn_units, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        return self.linear2(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ffn_units, dropout_rate):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, ffn_units, dropout_rate)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        attn_output = self.attention(x, x, x, mask)
        out1 = self.layernorm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout(ffn_output))

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ffn_units, dropout_rate):
        super(DecoderLayer, self).__init__()
        self.attention1 = MultiHeadAttention(d_model, num_heads)
        self.attention2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, ffn_units, dropout_rate)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        attn1 = self.attention1(x, x, x, look_ahead_mask)
        out1 = self.layernorm1(x + self.dropout(attn1))

        attn2 = self.attention2(out1, enc_output, enc_output, padding_mask)
        out2 = self.layernorm2(out1 + self.dropout(attn2))

        ffn_output = self.ffn(out2)
        return self.layernorm3(out2 + self.dropout(ffn_output))

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, ffn_units, num_layers, dropout_rate, max_len):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, ffn_units, dropout_rate)
            for _ in range(num_layers)
        ])

        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, ffn_units, dropout_rate)
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(d_model, vocab_size)

    def create_look_ahead_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask == 1

    def forward(self, encoder_input, decoder_input, encoder_mask=None, decoder_mask=None):
        # Encoder
        encoder_embedded = self.embedding(encoder_input)
        encoder_embedded = self.positional_encoding(encoder_embedded)

        encoder_output = encoder_embedded
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, encoder_mask)

        # Decoder
        decoder_embedded = self.embedding(decoder_input)
        decoder_embedded = self.positional_encoding(decoder_embedded)

        look_ahead_mask = self.create_look_ahead_mask(decoder_input.size(1)).to(decoder_input.device)

        decoder_output = decoder_embedded
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output, look_ahead_mask, encoder_mask)

        return self.fc(decoder_output)

class CustomTokenizer:
    def __init__(self, vocab_size=8000):
        self.vocab_size = vocab_size
        self.word2idx = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.word_count = {}

    def fit_on_texts(self, texts):
        for text in texts:
            for word in text.split():
                self.word_count[word] = self.word_count.get(word, 0) + 1
        sorted_vocab = sorted(self.word_count.items(), key=lambda x: x[1], reverse=True)[:self.vocab_size - 4]
        for idx, (word, _) in enumerate(sorted_vocab, start=4):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            seq = [self.word2idx.get(word, self.word2idx["<unk>"]) for word in text.split()]
            sequences.append(seq)
        return sequences

    def sequences_to_texts(self, sequences):
        texts = []
        for seq in sequences:
            text = " ".join([self.idx2word.get(idx, "<unk>") for idx in seq])
            texts.append(text)
        return texts

class WyckoffModelHandler:
    """
    Handler for loading and inference with your custom Transformer model
    """
    def __init__(self, model_path="wyckoff_model.pth", data_path='Cleaned_Wyckoff_QA_Dataset.csv'):
        # Hyperparameters matching your model
        self.MAX_LEN = 40
        self.NUM_HEADS = 8
        self.D_MODEL = 512
        self.FFN_UNITS = 2048
        self.DROPOUT = 0.1
        self.NUM_LAYERS = 6
        self.VOCAB_SIZE = 8000
        
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
    
    def _load_data_and_create_tokenizer(self):
        """Load the CSV data and create tokenizer"""
        try:
            # Try to load the original CSV data
            logger.info(f"Loading data from: {self.data_path}")
            data = pd.read_csv(self.data_path)
            data.columns = data.columns.str.strip()
            
            questions = data['Question'].astype(str).tolist()
            answers = data['Answer'].astype(str).tolist()
            
            logger.info(f"Creating tokenizer from {len(questions)} QA pairs")
            self.tokenizer = CustomTokenizer(self.VOCAB_SIZE)
            self.tokenizer.fit_on_texts(questions + answers)
            return True
        except Exception as e:
            logger.error(f"Error loading data and creating tokenizer: {e}")
            
            # Fallback to a minimal tokenizer
            logger.info("Creating minimal tokenizer with default vocabulary")
            self.tokenizer = CustomTokenizer(self.VOCAB_SIZE)
            
            # Add some basic Wyckoff terminology to ensure at least basic functionality
            basic_vocab = [
                "wyckoff", "method", "spring", "upthrust", "accumulation", "distribution", 
                "markup", "markdown", "volume", "price", "action", "composite", "man", 
                "effort", "result", "supply", "demand", "cause", "effect", "test"
            ]
            
            # Create a minimal dataset with these words
            minimal_texts = [" ".join(basic_vocab)]
            self.tokenizer.fit_on_texts(minimal_texts)
            return False
    
    def load_model(self):
        """Initialize the model architecture and load the state dict"""
        try:
            # First, make sure we have a tokenizer
            if self.tokenizer is None:
                success = self._load_data_and_create_tokenizer()
                if not success:
                    logger.warning("Using minimal tokenizer, responses may be limited")
            
            if os.path.exists(self.model_path):
                logger.info(f"Loading model from: {self.model_path}")
                
                # Initialize model with the same architecture used during training
                self.model = Transformer(
                    vocab_size=self.VOCAB_SIZE,
                    d_model=self.D_MODEL,
                    num_heads=self.NUM_HEADS,
                    ffn_units=self.FFN_UNITS,
                    num_layers=self.NUM_LAYERS,
                    dropout_rate=self.DROPOUT,
                    max_len=self.MAX_LEN
                ).to(self.device)
                
                # Load the saved state dictionary
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                
                # Set model to evaluation mode
                self.model.eval()
                
                logger.info(f"Model loaded successfully")
                return True
            else:
                logger.warning(f"Model file not found: {self.model_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            return False
    
    def generate_response(self, question, max_tokens=40):
        """Generate a response to a Wyckoff-related question"""
        if not self.model or not self.tokenizer:
            if not self.load_model():
                return None
                
        try:
            logger.info(f"Generating response for: {question}")
            
            # Preprocess the question
            question_seq = self.tokenizer.texts_to_sequences([question])[0]
            question_seq = [1] + question_seq[:self.MAX_LEN - 2] + [2]  # Add <start> and <end> tokens
            question_seq = question_seq + [0] * (self.MAX_LEN - len(question_seq))  # Pad to max_len
            question_tensor = torch.tensor([question_seq]).to(self.device)
            
            # Initialize decoder input with start token
            decoder_input = torch.tensor([[1]]).to(self.device)
            response = []
            
            # Generate response token by token
            for _ in range(max_tokens):
                with torch.no_grad():
                    output = self.model(question_tensor, decoder_input)
                
                # Get the most likely next token
                predicted_id = torch.argmax(output[:, -1, :], dim=-1).item()
                
                # Stop if end token is generated
                if predicted_id == 2:  # End token
                    break
                    
                response.append(predicted_id)
                
                # Add the predicted token to decoder input for next iteration
                decoder_input = torch.cat([decoder_input, torch.tensor([[predicted_id]]).to(self.device)], dim=-1)
            
            # Convert token ids back to text
            response_text = self.tokenizer.sequences_to_texts([response])[0]
            
            # Clean up the response
            response_text = response_text.replace("<start>", "").replace("<end>", "").strip()
            logger.info(f"Generated response: {response_text[:50]}...")
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return None
                
    def get_fallback_response(self, question=None):
        """Generate a fallback response when model is not available"""
        wyckoff_concepts = [
            "Wyckoff's methodology is based on the analysis of price action, volume, and time.",
            "The Wyckoff Method consists of three fundamental laws: Supply and Demand, Cause and Effect, and Effort vs Result.",
            "Wyckoff identified four market phases: Accumulation, Markup, Distribution, and Markdown.",
            "In the Wyckoff Method, 'springs' occur when price briefly penetrates support in an accumulation phase.",
            "Wyckoff's 'Composite Man' concept represents the collective actions of large institutional investors.",
            "The Wyckoff method helps traders identify the direction of the larger trend and the most probable direction of future price action.",
            "Wyckoff's 'tests' occur when price returns to a previous support or resistance level to confirm its strength.",
            "Volume analysis is critical in the Wyckoff method to confirm price movements and identify potential reversals.",
            "Wyckoff emphasized the importance of trading in harmony with the larger market cycle and trend."
        ]
        
        # If question contains specific Wyckoff terms, provide more targeted responses
        question_lower = question.lower() if question else ""
        
        if "upthrust" in question_lower:
            return "An upthrust in Wyckoff methodology is a price move that penetrates above a resistance level, often on increased volume, and then quickly reverses. It's typically a sign of distribution as smart money sells into retail buying. Upthrusts often trap buyers at higher prices and signal potential trend reversal or continuation of a downtrend."
        
        elif "spring" in question_lower:
            return "A spring in Wyckoff methodology is when price briefly penetrates below a support level and then quickly reverses upward. It represents the final test of a support level before markup begins. Springs often trap sellers at lower prices and signal that smart money has completed its accumulation phase."
        
        elif "accumulation" in question_lower:
            return "Accumulation in Wyckoff methodology is a phase where smart money begins buying an asset after a prolonged downtrend. This phase is characterized by decreased volatility, testing of support levels, springs, and increased trading range. The goal of accumulation is for smart money to acquire positions before markup begins."
        
        elif "distribution" in question_lower:
            return "Distribution in Wyckoff methodology is a phase where smart money begins selling their accumulated positions to the public after a prolonged uptrend. This phase features decreased volatility, testing of resistance levels, upthrusts, and signs of supply overwhelming demand. It often precedes a price decline."
        
        # Default to random concept if no specific terms are found
        import random
        return f"Based on Wyckoff analysis: {random.choice(wyckoff_concepts)}"