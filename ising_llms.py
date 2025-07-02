import numpy as np
from typing import List, Tuple
import yaml
import os
import matplotlib.pyplot as plt 
import random
import torch
import time


from openai import OpenAI
from groq import Groq
from transformers import AutoModelForCausalLM, AutoTokenizer
import ollama
import yaml

# Load the config file
with open('config.yaml', 'r') as file: 
    config = yaml.safe_load(file)


############ INITIALIZE SEED #############
random.seed(config['seed'])
np.random.seed(config['seed'])
torch.manual_seed(config['seed']) 


if config['service']=='groq':
    model_name = config['model']
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=GROQ_API_KEY)
elif config['service']=='openai':
    model_name = config['model']
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=OPENAI_API_KEY)
elif config['service']=='huggingface':
    model_name = config['model']
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    client = {"model": model_name, "tokenizer": tokenizer}
elif config['service']=='ollama':
    model_name = config['model']
else:
    print('Select a client!')


# Load prompts from prompts.txt
def load_prompts():
    prompts = {}
    with open('prompts_list.txt', 'r') as file:
        content = file.read()
        # Split by 'prompt_' and process each section
        sections = content.split('prompt_')
        for section in sections[1:]:  # Skip first empty section
            lines = section.strip().split('\n')
            prompt_name = lines[0].split('=')[0].strip()
            prompt_text = '\n'.join(lines[1:]).strip().strip('"""')
            prompts[prompt_name] = prompt_text
    return prompts

PROMPTS = load_prompts()

class LLMSpin:
    def __init__(self, position: Tuple[int, int], initial_state: str = "neutral"):
        self.position = position
        self.state = initial_state # random.choice(['yes','no'])
        if config['service']=='huggingface' or config['service']=='openai' or config['service']=='groq':
            self.client = client
        elif config['service']=='ollama':
            self.client = None
        #self.memory = []  # Store interaction history

        
    def get_response(self, neighbors_states: List[str], temperature: float) -> str:
        """Generate LLM response based on neighbor states and temperature."""
        # Use selected prompt from prompts.txt
        prompt_template = PROMPTS[config['prompt_type']]
        formatted_prompt = prompt_template.format(
            state=self.state,
            neighbors_states=neighbors_states
        )

        try:
            if config['service']=='huggingface':
                # Handle Hugging Face model response
                inputs = self.client["tokenizer"](formatted_prompt, return_tensors="pt")
                # HF output contains the prompt and the new tokens
                outputs = self.client["model"].generate(
                    inputs["input_ids"],
                    max_new_tokens=config['max_tokens'],
                    temperature=temperature,
                    do_sample=True
                )
                # compute input length in tokens
                prompt_len = inputs["input_ids"].shape[-1]
                # slice off the prompt part, keeping only the new tokens
                new_tokens = outputs[0][prompt_len:]
                new_state = self.client["tokenizer"].decode(new_tokens, skip_special_tokens=True).strip().lower()
            elif config['service']=='openai' or config['service']=='groq':
                # Handle OpenAI/Groq response
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "system", "content": formatted_prompt}],
                    temperature=temperature,
                    max_tokens=config['max_tokens'],
                )
                new_state = response.choices[0].message.content.strip().lower()
            elif config['service']=='ollama':
                response = ollama.generate(model=model_name, 
                         prompt=formatted_prompt, 
                         options={"temperature":temperature, "num_predict": config['max_tokens']})
                new_state = response['response']
            #print(new_state)
            #time.sleep(0.65)
            #return 'agree' if 'agree' in new_state else 'disagree'
            return 'yes' if 'yes' in new_state else 'no'
        except Exception as e:
            print(f"Error in LLM response: {e}")
            return self.state

class IsingLLM:
    def __init__(self, size: int, temperature: float = 0.5):
        self.size = size
        self.temperature = temperature
        # We init the lattice in a random config of yes and no states
        self.lattice = [[LLMSpin((i, j), random.choice(['yes','no'])) for j in range(size)] for i in range(size)]
        #self.lattice = [[LLMSpin((i, j), 'neutral') for j in range(size)] for i in range(size)]
        
        # Update file naming to include prompt type
        self.output_file = f"magnetizations/isingllm_magnetization__seed-{config['seed']}_model-{config['model']}_size-{self.size}_temp-{self.temperature}_prompt-{config['prompt_type']}.csv"
        #self.output_file = f"magnetizations/isingllm_magnetization__seed-{config['seed']}_model-llama-4-scout-17b-16e-instruct_size-{self.size}_temp-{self.temperature}.csv"
        
        # Initialize magnetization file
        with open(self.output_file, 'w') as f:
            f.write("Step, Magnetization, M**2, M**4\n")
        
    def get_neighbors(self, i: int, j: int) -> List[str]:
        """Get states of neighboring spins (periodic boundary conditions)."""
        neighbors = []
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni, nj = (i + di) % self.size, (j + dj) % self.size
            neighbors.append(self.lattice[ni][nj].state)
        return neighbors
    
    def update_site(self, i: int, j: int):
        """Update single site based on neighbors."""
        neighbors_states = self.get_neighbors(i, j)
        new_state = self.lattice[i][j].get_response(neighbors_states, self.temperature)
        self.lattice[i][j].state = new_state
        
    def get_lattice_array(self) -> np.ndarray:
        """Convert current lattice state to numpy array."""
        return np.array([[1 if self.lattice[i][j].state == 'yes' else 0 
                         for j in range(self.size)] 
                        for i in range(self.size)])

    def step(self):
        """Perform one Monte Carlo step."""
        # Randomly select sites to update
        for _ in range(self.size * self.size):
            i, j = np.random.randint(0, self.size), np.random.randint(0, self.size)
            self.update_site(i, j)
    
    def calculate_magnetization(self) -> float:
        """Calculate system magnetization."""
        agree_count = sum(1 for i in range(self.size) for j in range(self.size) 
                         if self.lattice[i][j].state == 'yes')
        return (2 * agree_count - self.size * self.size) / (self.size * self.size)
    
    def visualize(self):
        """Visualize current lattice state."""
        for i in range(self.size):
            row = []
            for j in range(self.size):
                symbol = '↑' if self.lattice[i][j].state == 'yes' else '↓'
                row.append(symbol)
            print(' '.join(row))
        print()

    def save_lattice_state(self, step: int):
        """Save the current lattice state to a numpy array file."""
        lattice_array = self.get_lattice_array()
        step_filename = f"lattices/lattice-state__step_{step}_seed-{config['seed']}_model-{config['model']}_size-{self.size}_temp-{self.temperature}_prompt-{config['prompt_type']}.npy"
        #step_filename = f"lattices/lattice-state__step_{step}_seed-{config['seed']}_model-llama-4-scout-17b-16e-instruct_size-{self.size}_temp-{self.temperature}.npy"
        np.save(step_filename, lattice_array)

    def save_lattice_image(self, step: int = None):
        """Save current lattice state as a PNG image of black and white dots."""
        # Create a binary matrix (1 for 'yes'/up, 0 for 'no'/down)
        #lattice_array = np.array([[1 if self.lattice[i][j].state == 'yes' else 0 
        #                         for j in range(self.size)] 
        #                        for i in range(self.size)])
        lattice_array = self.get_lattice_array()
        
        # Create figure and axis
        plt.figure(figsize=(8, 8))
        plt.imshow(lattice_array)
        plt.grid(True, color='gray', linestyle='-', linewidth=0.5)
        
        # Remove axes ticks
        plt.xticks([])
        plt.yticks([])
        
        # Create filename
        filename = f"images/lattice-image"
        if step is not None:
            filename += f"_step{step}"
        filename += f"__seed-{config['seed']}_model-{config['model']}_size-{self.size}_temp-{self.temperature}_prompt-{config['prompt_type']}.png"
        #filename += f"__seed-{config['seed']}_model-llama-4-scout-17b-16e-instruct_size-{self.size}_temp-{self.temperature}.png"

        # Save the figure
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close()
        
        #print(f"Lattice state image saved as {filename}")

    def save_step_magnetization(self, step: int, magnetization: float):
        """Save magnetization for current step immediately to file."""
        with open(self.output_file, 'a') as f:
            f.write(f"{step}, {magnetization:.5f}, {magnetization**2:.5f}, {magnetization**4:.5f}\n")