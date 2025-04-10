import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from jaxtyping import Bool, Float, Int
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List

warnings.filterwarnings("ignore")

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True  # Mark end of word

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False  # Word not found
            node = node.children[char]
        return node.is_end_of_word  # Returns True if it's a complete word

class ConstrainedTextGenerator:
    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        eos_id: int, 
        max_output_len: int = 10,
    ) -> None:
        '''
            Initialize the ConstrainedTextGenerator class.
            
            model: LLM
            tokenizer: LLM's tokenizer.
            eos_id: End-of-sequence token id 
            max_output_len: Maximum number of tokens to be generated.
            
            Do not edit.
        '''
        self.model = model
        
        self.max_output_len = max_output_len
        self.eos_token_id = eos_id
        
        self.tokenizer = tokenizer

    def __call__(
        self, input_ids: Int[torch.Tensor, "batch in_seq_len"], word_list: list
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Implement Word-Constrained decoding technique. (refer assignment document for more details)
            
            `word_list`: contains bag of words for the particular example

            - batch size is always 1; no need to handle generation of multiple sequences simultaneously.
            - stop decoding when: 
                - the end-of-sequence (EOS) token is generated `self.eos_token_id`
                - the predefined maximum number of tokens is reached `self.max_output_len`
            
            Return an integer tensor containing only the generated tokens (excluding input tokens).
            
            Input:
                input_ids: tensor of shape (1, P)
            Returns:
                tensor of shape (T,), where T <= self.max_output_len
        '''    
        input_id = input_ids.clone().to("cuda")  
        # print("1.1")
        trie = Trie()
        for word in word_list:
            trie.insert(word)

        while input_id.shape[1] < (input_ids.shape[1] + self.max_output_len):  
            logits = self.model(input_id).logits  
            last_token_logits = logits[:, -1, :] 
            prob = F.softmax(last_token_logits, dim=-1).to('cuda')  
            
            sorted_tokens = torch.argsort(prob, descending=True) 

            for token in sorted_tokens[0]:
                decoded_word = self.tokenizer.decode(token.cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=True)

                if trie.search(decoded_word):
                    next_token = token  # Found a valid token
                    break
            else:
                next_token = sorted_tokens[0, 0]

            next_token = torch.tensor(next_token).view(1)
            # print(f"shape of input_is: {input_id.shape}")
            # print(f"shape of n_token is: {next_token.shape}")
            # print(f"max len is: {input_ids.shape[1] + self.max_output_len}")
            input_id = torch.cat((input_id, next_token.unsqueeze(0)), dim=1)  

            if next_token.item() == self.eos_token_id:
                break  

        return input_id[0, input_ids.shape[1]:]

