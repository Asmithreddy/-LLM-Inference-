import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from jaxtyping import Bool, Float, Int
from transformers import AutoModelForCausalLM
from typing import List

warnings.filterwarnings("ignore")

class TextGenerator:
    def __init__(
        self, 
        model: AutoModelForCausalLM,                                                              
        decoding_strategy: str, 
        eos_id: int,     
        max_output_len: int = 10,
        tau: int = 1,
        k: int = 10,
        p: int = 0.5
    ) -> None:
        '''
            Initialize the TextGenerator class.             
            
            model: LLM  
            decoding_strategy: str describing the decoding strategy to be used.
            eos_id: End-of-sequence token id 
            max_output_len: Maximum number of tokens to be generated. 
            tau: Temperature parameter for random sampling 
            k: Top-k parameter for top-k sampling 
            p: Cumulative probability threshold for nucleus  
            
            Do not edit.
        '''
        self.model = model 
        self.decoding_strategy = decoding_strategy 
        
        self.max_output_len = max_output_len
        self.eos_token_id = eos_id
        self.tau = tau 
        self.k = k 
        self.p = p
        
        if decoding_strategy == "greedy":
            self.generator_func = self.greedy_decoding   
        elif decoding_strategy == "random":
            self.generator_func = self.random_sampling
        elif decoding_strategy == "topk":
            self.generator_func = self.topk_sampling
        elif decoding_strategy == "nucleus":
            self.generator_func = self.nucleus_sampling

    def __call__(
        self, 
        input_ids: Int[torch.Tensor, "batch in_seq_len"], 
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Do not edit.
        '''
        return self.generator_func(input_ids)
                
    def greedy_decoding(
        self,
        input_ids: Int[torch.Tensor, "batch in_seq_len"], 
    ) -> Int[torch.Tensor, "batch out_seq_len"]: 
        '''
            Implement Greedy decoding technique. (refer assignment document for more details)

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

        while input_id.shape[1] < (input_ids.shape[1] + self.max_output_len):  
            logits = self.model(input_id).logits 
            last_token_logits = logits[:, -1, :]  
            prob = F.softmax(last_token_logits, dim=-1).to('cuda')  
            next_token = torch.argmax(prob, dim=-1)  

            input_id = torch.cat((input_id, next_token.unsqueeze(0)), dim=1)  

            if next_token.item() == self.eos_token_id:
                break  

        return input_id[0, input_ids.shape[1]:]

        
    def random_sampling(
        self, 
        input_ids: Int[torch.Tensor, "batch in_seq_len"]   
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Implement Random sampling technique. (refer assignment document for more details)

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
        tau=self.tau
        while input_id.shape[1] < (input_ids.shape[1] + self.max_output_len):  
            logits = self.model(input_id).logits  
            last_token_logits = logits[:, -1, :]  
            prob = F.softmax(last_token_logits, dim=-1).to('cuda')  

            prob = torch.pow(prob, 1/tau)  
            prob = prob / prob.sum(dim=-1, keepdim=True)  

            sampled_token = torch.multinomial(prob, num_samples=1)
            input_id = torch.cat((input_id, sampled_token), dim=1)  

            if sampled_token.item() == self.eos_token_id:
                break  

        return input_id[0, input_ids.shape[1]:]
    
    def topk_sampling(
        self, 
        input_ids: Int[torch.Tensor, "batch in_seq_len"]
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Implement Top-k sampling technique. (refer assignment document for more details)

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
        k=self.k   
        while input_id.shape[1] < (input_ids.shape[1] + self.max_output_len):  
            logits = self.model(input_id).logits  
            last_token_logits = logits[:, -1, :]   
            prob = F.softmax(last_token_logits, dim=-1).to('cuda')  

            top_k_probs, top_k_indices = torch.topk(prob, k, dim=-1) 
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  
            sampled_index = torch.multinomial(top_k_probs, num_samples=1) 
            sampled_token = top_k_indices.gather(dim=-1, index=sampled_index) 

            input_id = torch.cat((input_id, sampled_token), dim=1)  

            if sampled_token.item() == self.eos_token_id:
                break  

        return input_id[0, input_ids.shape[1]:]
    
    def nucleus_sampling(
        self, 
        input_ids: Int[torch.Tensor, "batch in_seq_len"]
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Implement Nucleus sampling technique. (refer assignment document for more details)

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
        p=self.p    # p={0.5,0.9}
        while input_id.shape[1] < (input_ids.shape[1] + self.max_output_len):  
            logits = self.model(input_id).logits 
            last_token_logits = logits[:, -1, :]  
            prob = F.softmax(last_token_logits, dim=-1).to('cuda')

            sorted_probs, sorted_indices = torch.sort(prob, descending=True, dim=-1)

            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            cutoff_index = (cumulative_probs > p).nonzero(as_tuple=True)[1].min().item()
            top_p_probs = sorted_probs[:, :cutoff_index + 1]
            top_p_indices = sorted_indices[:, :cutoff_index + 1]

            top_p_probs /= top_p_probs.sum(dim=-1, keepdim=True) 

            sampled_index = torch.multinomial(top_p_probs, num_samples=1)  
            sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

            input_id = torch.cat((input_id, sampled_token), dim=1)  

            if sampled_token.item() == self.eos_token_id:
                break  

        return input_id[0, input_ids.shape[1]:]

        