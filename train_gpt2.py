from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size : int = 50257
    n_layer: int = 12
    n_head:int = 12
    n_embd : int = 768

class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd,3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd,config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size)).view(1,1,config.block_size,config.block_size))
    
    def forward(self, x):
        B , T , C = x.size() # Batch size , Sequence length , embedding dimesionality(n_embd)
        #nh -> "numbe of heads" , hs -> "head size" , c -> "number of channels"
        qkv = self.c_attn(x)
        q , k , v= qkv.split(self.n_embd,dim=2)
        k = k.view(B , T , self.n_head , C // self.n_head).transpose(1,2) #(B,nh,T,hs)
        q = q.view(B , T , self.n_head , C // self.n_head).transpose(1,2) #(B,nh,T,hs)
        v = v.view(B , T , self.n_head , C // self.n_head).transpose(1,2) #(B,nh,T,hs)

        att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att,dim=-1)
        y = att @ v
        y =  y.transpose(1,2).contiguous().view(B , T , C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd,4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd , config.n_embd)
    
    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size , config.n_embd),
                wpe = nn.Embedding(config.block_size , config.n_embd),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = nn.LayerNorm(config.n_embd),
            ))
        self.lm_head = nn.Linear(config.n_embd,config.vocab_size,bias = False)
    
    def forward(self,idx , targets = None):
        B,T = idx.size()
        assert T <= self.config.block_size , f"Cannot forward sequence of lenght {T},block size is {self.config.block_size}"
        
        pos = torch.arange(0 , T , dtype=torch.long , device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))
        
        return logits , loss
                    
    
    @classmethod
    def from_pretrained(cls,model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        
        assert model_type in {"gpt2","gpt2-medium","gpt2-large","gpt2-xl"}
        
        from transformers import GPT2LMHeadModel
        print("Loading weights from pretrained gpt : %s"%model_type)
        
        config_args = {
            "gpt2" : dict(n_layer=12,n_head=12,n_embd=768), #124M
            "gpt2-medium" : dict(n_layer=24,n_head=16,n_embd=1024), #350M
            "gpt2-large" : dict(n_layer=36,n_head=20,n_embd=1280), #774M
            "gpt2-medium" : dict(n_layer=48,n_head=25,n_embd=1600), #1558M
        }[model_type]
        
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024
        
        config = GPTConfig(**config_args)
        model = GPT(config=config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]
        
        # init a huggingface transformer model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight','attn.c_proj.weight','mlp.c_fc.weight','mlp.c_proj.weight']
        
        assert len(sd_keys_hf) == len(sd_keys) , f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
    

#_____________________________________________________________

import tiktoken

class DataLoaderLite:
    def __init__(self,B,T):
        self.B = B
        self.T = T
        
        with open("input.txt","r") as f:
            text = f.read()
        
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B*T)} batches")
        
        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B , self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B,T)
        y = (buf[1:]).view(B,T)
        self.current_position += B * T
        
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x ,y
        
#_______________________________

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print(f"using device : {device}")

train_loader = DataLoaderLite(B=4,T=32)

# get logits 
model = GPT(GPTConfig())
model.to(device)

#optimization
optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4)
epochs = 100
for i in range(epochs):
    x , y = train_loader.next_batch()
    x , y = x.to(device) , y.to(device)
    optimizer.zero_grad()
    logits,loss= model(x,y)
    loss.backward()
    optimizer.step()
    print(f"Step {i} , loss: {loss.item()}")


import sys; sys.exit(0)



num_return_sequences = 5
max_length = 30
# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig())
model.eval()
model.to(device)

import tiktoken
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello I am language model,")
tokens = torch.tensor(tokens,dtype=torch.long) #(8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences,1) #(5 , 8)
x = tokens.to(device)
        
torch.manual_seed(42)
torch.cuda.manual_seed(42)

while x.size(1) < max_length:
    
    with torch.no_grad():
        logits = model(x) #(B, T , vocab_size)
        logits = logits[:,-1,:] #(B , vocab_size)
        probs = F.softmax(logits,dim=-1)
        
        topk_probs , topk_indices = torch.topk(probs , 50 , dim=-1)
        
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices,-1,ix)
        
        x = torch.cat((x,xcol),dim=1)
        
# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, : max_length].tolist()
    decoded = enc.decode(tokens)
    print(">",decoded)