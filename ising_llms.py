import numpy as np
from typing import List, Tuple
#import yaml
#import argparse
#import os
import os
import matplotlib.pyplot as plt 
import random
import torch
#import time

#from openai import OpenAI
#from groq import Groq
#from transformers import AutoModelForCausalLM, AutoTokenizer
import ollama


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
    def __init__(self, position: Tuple[int, int], initial_state: str = "neutral", client: dict = None, global_flip: int = 1):
        self.position = position
        self.state = initial_state 
        self.global_flip = global_flip
        self.client = client
        self.model_name = client['model']
        self.service = client['service']

        if self.service=='huggingface':
            self.client = client
        elif self.service=='openai' or  self.service=='groq':
            self.client = client["client"]
        elif self.service=='ollama':
            self.client = client["client"] # which is None

    def get_response(self, neighbors_states: List[str], temperature: float, max_tokens: int = 1, prompt_type: str = None) -> str:
        """Generate LLM response based on neighbor states and temperature."""
        # Use selected prompt from prompts.txt
        if prompt_type in ['nr6', 'nr7', 'nr8']:
            if neighbors_states.count('yes') == 3:
                neighbors_states = 'majority yes'
            elif neighbors_states.count('yes') == 2:
                neighbors_states = 'half yes and half no' if random.random() < 0.5 else 'half no and half yes'
            else:
                neighbors_states = 'majority no'
            modality = "'yes' or 'no'" if random.random() < 0.5 else "'no' or 'yes'"

        prompt_template = PROMPTS[prompt_type]
        formatted_prompt = prompt_template.format(
            state=self.state,
            neighbors_states=neighbors_states,
            modality=modality
        )

        #print("Formatted prompt:", formatted_prompt)

        try:
            if self.service=='huggingface':
                # Handle Hugging Face model response
                inputs = self.client["tokenizer"](formatted_prompt, return_tensors="pt")
                # HF output contains the prompt and the new tokens
                outputs = self.client["model"].generate(
                    inputs["input_ids"],
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True
                )
                # compute input length in tokens
                prompt_len = inputs["input_ids"].shape[-1]
                # slice off the prompt part, keeping only the new tokens
                new_tokens = outputs[0][prompt_len:]
                raw_state = self.client["tokenizer"].decode(new_tokens, skip_special_tokens=True).strip().lower()
            elif self.service=='openai' or self.service=='groq':
                # Handle OpenAI/Groq response
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "system", "content": formatted_prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                raw_state = response.choices[0].message.content.strip().lower()
            elif self.service=='ollama':
                response = ollama.generate(model=self.model_name, 
                         prompt=formatted_prompt, 
                         options={"temperature": temperature, "num_predict": max_tokens})
                raw_state = response['response'].strip().lower()
            #print("Raw state:", raw_state)
            #time.sleep(0.65)

            # ------- global-flip post-processing ----------------------
            # this ensures that the system is unbiased and enforces z_2 symmetry
            sigma_raw   = 1 if 'yes' in raw_state else -1    
            sigma_final = self.global_flip * sigma_raw      
            #print("Sigma raw:", sigma_raw)
            #print("Sigma final:", sigma_final)
            new_state = 'yes' if sigma_final == 1 else 'no'
            #print("New state:", new_state)
            return new_state
        except Exception as e:
            print(f"Error in LLM response: {e}")
            return self.state

class IsingLLM:
    def __init__(self, size: int, temperature: float = 0.5, client: dict = None, random_seed: int = None, 
                 prompt_type: str = None, max_tokens: int = 1, z2_symmetry: bool = False, ties_fifty: bool = False):
        self.client = client
        self.model_name = client['model']
        self.size = size
        self.temperature = temperature
        self.random_seed = random_seed
        self.prompt_type = prompt_type
        self.max_tokens = max_tokens
        self.z2_symmetry = z2_symmetry
        self.ties_fifty = ties_fifty
        #with open('config.yaml', 'r') as file: 
        #    config = yaml.safe_load(file)

        random.seed(self.random_seed)    
        np.random.seed(self.random_seed) 
        torch.manual_seed(self.random_seed) 
        # choose global flip once (+ or - 1): to unbias the system and enforce z_2 symmetry
        if self.z2_symmetry == True:
            self.global_flip  = 1 if np.random.rand() < 0.5 else -1
            # We init the lattice in a random config of yes and no states
            self.lattice = [[LLMSpin((i, j), random.choice(['yes','no']), self.client, self.global_flip) for j in range(size)] for i in range(size)]
            print(f"Global flip (Z2: true): {self.global_flip}")
        else:
            # We init the lattice in a random config of yes and no states
            self.global_flip  = 1
            self.lattice = [[LLMSpin((i, j), random.choice(['yes','no']), self.client, self.global_flip) for j in range(size)] for i in range(size)]
            print(f"Global flip (Z2: false): {self.global_flip}")
         
        # Update file naming to include prompt type
        model_name_save = self.model_name.replace(":", "-")
        self.output_file = f"magnetizations/isingllm_magnetization__seed-{self.random_seed}_model-{model_name_save}_size-{self.size}_temp-{self.temperature}_prompt-{self.prompt_type}.csv"
        #self.output_file = f"magnetizations/isingllm_magnetization__seed-{config['seed']}_model-llama-4-scout-17b-16e-instruct_size-{self.size}_temp-{self.temperature}.csv"
        
        # Initialize magnetization file
        with open(self.output_file, 'w') as f:
            f.write("Step, Magnetization, M**2, M**4\n")
        
        # Initialize per-site updates CSV
        os.makedirs('site_updates', exist_ok=True)
        self.site_updates_file = (
            f"site_updates/site-updates__seed-{self.random_seed}_model-{model_name_save}_"
            f"size-{self.size}_temp-{self.temperature}_prompt-{self.prompt_type}.csv"
        )
        with open(self.site_updates_file, 'w') as f:
            f.write("step, i,j,neighbors_states[0],neighbors_states[1],neighbors_states[2],neighbors_states[3], old_state, new_state\n")
        
    def get_neighbors(self, i: int, j: int) -> List[str]:
        """Get states of neighboring spins (periodic boundary conditions)."""
        neighbors = []
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni, nj = (i + di) % self.size, (j + dj) % self.size
            neighbors.append(self.lattice[ni][nj].state)
        return neighbors
    
    def update_site(self, i: int, j: int, step: int,):
        """Update single site based on neighbors."""
        old_state = self.lattice[i][j].state
        neighbors_states = self.get_neighbors(i, j)
        if self.ties_fifty == False:
            new_state = self.lattice[i][j].get_response(neighbors_states, self.temperature, self.max_tokens, self.prompt_type)
            #print("NS: ", new_state)
        elif neighbors_states.count('yes') == 2 and self.ties_fifty == True:
            new_state = "yes" if np.random.rand() < 0.5 else "no"
            #print("NS ---: ", new_state)
        elif neighbors_states.count('yes') != 2 and self.ties_fifty == True:
            new_state = self.lattice[i][j].get_response(neighbors_states, self.temperature, self.max_tokens, self.prompt_type)
            #print("NS: ", new_state)
        self.lattice[i][j].state = new_state
        # persist per-site update immediately
        self.save_site_update(step, i, j, neighbors_states, old_state, new_state)
        
    def get_lattice_array(self) -> np.ndarray:
        """Convert current lattice state to numpy array."""
        return np.array([[1 if self.lattice[i][j].state == 'yes' else 0 
                         for j in range(self.size)] 
                        for i in range(self.size)])

    def step(self, step: int):
        """Perform one Monte Carlo step."""
        # Randomly select sites to update
        for _ in range(self.size * self.size):
            i, j = np.random.randint(0, self.size), np.random.randint(0, self.size)
            self.update_site(i, j, step)
    
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
        model_name_save = self.model_name.replace(":", "-")
        step_filename = f"lattices/lattice-state__step_{step}_seed-{self.random_seed}_model-{model_name_save}_size-{self.size}_temp-{self.temperature}_prompt-{self.prompt_type}.npy"
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
        model_name_save = self.model_name.replace(":", "-")
        filename += f"__seed-{self.random_seed}_model-{model_name_save}_size-{self.size}_temp-{self.temperature}_prompt-{self.prompt_type}.png"
        #filename += f"__seed-{config['seed']}_model-llama-4-scout-17b-16e-instruct_size-{self.size}_temp-{self.temperature}.png"

        # Save the figure
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close()
        
        #print(f"Lattice state image saved as {filename}")

    def save_step_magnetization(self, step: int, magnetization: float):
        """Save magnetization for current step immediately to file."""
        with open(self.output_file, 'a') as f:
            f.write(f"{step}, {magnetization:.8f}, {magnetization**2:.8f}, {magnetization**4:.8f}\n")

    def save_site_update(self, step: int, i: int, j: int, neighbors_states: List[str], old_state: str, new_state: str) -> None:
        """Append one site update row: i,j, neighbors[0..3], old_state, new_state."""
        with open(self.site_updates_file, 'a') as f:
            f.write(
                f"{step},{i},{j},{neighbors_states[0]},{neighbors_states[1]},{neighbors_states[2]},{neighbors_states[3]},{old_state},{new_state}\n"
            )