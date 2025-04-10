import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from jaxtyping import Bool, Float, Int
from transformers import AutoModelForCausalLM 
from typing import List

warnings.filterwarnings("ignore")

class MedusaTextGenerator:
    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        decoding_strategy: str, 
        eos_id: int, 
        use_no_medusa_heads: int = 5,
        beam_width: int = 2,
        max_output_len: int = 10,
    ) -> None:
        '''
            Initialize the MedusaTextGenerator class.
            
            model: LLM 
            decoding_strategy: str describing the decoding strategy to be used.
            eos_id: End-of-sequence token id 
            use_no_medusa_heads: Number of medusa heads to be used (maximum:5) (denoted as S).
            beam_width: Maximum number of candidates that can be present in the beam (denoted as W).
            max_output_len: Maximum number of tokens to be generated.
            
            Do not edit.
        '''
        self.model = model    
        self.decoding_strategy = decoding_strategy
        
        self.max_output_len = max_output_len
        self.eos_token_id = eos_id
        self.beam_width = beam_width   
        
        assert use_no_medusa_heads <= 5, "The current medusa model supports at max 5 heads"
        self.no_heads = use_no_medusa_heads + 1
        
        if decoding_strategy == "single-head":
            self.generator_func = self.single_head_decoding
        elif decoding_strategy == "multi-head":
            self.generator_func = self.multi_head_decoding
        
    def __call__(
        self, input_ids: Int[torch.Tensor, "batch in_seq_len"]
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Do not edit.
        '''
        return self.generator_func(input_ids)
                
    def single_head_decoding(
        self,
        input_ids: Float[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "batch out_seq_len"]:     
        '''
            Implement Single-head decoding technique. Use only LM head for decoding here (refer assignment document for more details)

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
        input_id=input_ids.clone().to('cuda')

        while(input_id.shape[1]<input_ids.shape[1]+self.max_output_len):
            medusa_logits, _, base_logits = self.model(input_id, medusa_forward=True, output_orig=True)
            prob=F.softmax(base_logits[:,-1,:],dim=-1).to('cuda')
            next_token=torch.argmax(prob,dim=-1)

            input_id=torch.concat((input_id,next_token.unsqueeze(0)),dim=-1)

            if(next_token.item() == self.eos_token_id):
                break
            
        return input_id[0,input_ids.shape[1]:]

    def multi_head_decoding(
        self,
        input_ids: Float[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "batch out_seq_len"]:     
        '''
            Implement multi-head decoding technique. (refer assignment document for more details)

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

        generated_tokens = []
        current_seq = input_ids.clone().to('cuda') 
        
        while len(generated_tokens) < self.max_output_len:

            medusa_logits, _, base_logits = self.model(current_seq, medusa_forward=True, output_orig=True)
            
            distributions = []
            distributions.append(base_logits[:, -1, :])  
            for k in range(self.no_heads - 1):  
                if k < len(medusa_logits):
                    distributions.append(medusa_logits[k][:, -1, :])
            
    
            candidates = [current_seq]
            scores = [0.0]
            
            for s in range(len(distributions)):
                log_p = F.log_softmax(distributions[s], dim=-1)
                new_candidates = []
                new_scores = []
                
                for c in range(len(candidates)):
                    topk_log_probs, topk_tokens = torch.topk(log_p, self.beam_width, dim=-1)
                    
                    for i in range(self.beam_width):
                        token = topk_tokens[0, i].unsqueeze(0).unsqueeze(0)
                        new_score = scores[c] + topk_log_probs[0, i].item()
                        new_seq = torch.cat([candidates[c], token], dim=-1)
                        
                        new_candidates.append(new_seq)
                        new_scores.append(new_score)
                
                if new_scores:
                    sorted_indices = sorted(range(len(new_scores)), key=lambda i: -new_scores[i])
                    candidates = [new_candidates[i] for i in sorted_indices[:self.beam_width]]
                    scores = [new_scores[i] for i in sorted_indices[:self.beam_width]]
            
            final_scores = []
            for candidate in candidates:
                with torch.no_grad():
                    outputs = self.model(candidate)
                    logits = outputs.logits
                
                score = 0.0
                for i in range(current_seq.shape[-1], candidate.shape[-1]):
                    log_p = F.log_softmax(logits[:, i-1, :], dim=-1)
                    token = candidate[0, i]
                    score += log_p[0, token].item()
                
                final_scores.append(score)
            
            best_idx = torch.argmax(torch.tensor(final_scores)).item()
            best_candidate = candidates[best_idx]
        
            new_tokens = best_candidate[0, current_seq.shape[-1]:]
            
            eos_positions = (new_tokens == self.eos_token_id).nonzero()
            if eos_positions.numel() > 0:
                first_eos = eos_positions[0].item()
                generated_tokens.extend(new_tokens[:first_eos+1].tolist())
                break
            
            generated_tokens.extend(new_tokens.tolist())
            current_seq = best_candidate
    
            if len(generated_tokens) >= self.max_output_len:
                break
        
        return torch.tensor(generated_tokens[:self.max_output_len], device=input_ids.device)


        
    




        

    
        




            