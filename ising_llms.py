import numpy as np
from typing import List, Tuple
import yaml
import os
import matplotlib.pyplot as plt 
import time
import random


from openai import OpenAI
from groq import Groq

# Load the config file
with open('config.yaml', 'r') as file: 
    config = yaml.safe_load(file)


############ INITIALIZE SEED #############
random.seed(config['seed'])
np.random.seed(config['seed'])
#torch.manual_seed(config['seed']) 



if config['groq_api']==True:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=GROQ_API_KEY)
elif config['openai_api']==True:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    print('Select a client!')


class LLMSpin:
    def __init__(self, position: Tuple[int, int], initial_state: str = "neutral"):
        self.position = position
        self.state = initial_state # random.choice(['yes','no'])
        self.client = client
        #self.memory = []  # Store interaction history

        
    def get_response(self, neighbors_states: List[str], temperature: float) -> str:
        """Generate LLM response based on neighbor states and temperature."""
        # Construct prompt considering neighbors and temperature
        prompt_standard = f"""
        You are a node in a 2D lattice system. Your current state is '{self.state}'.
        Your neighboring nodes have the following states: {neighbors_states}.
        Do you want to align with them or not? 
        Based on these conditions, you can only respond with either 'yes' or 'no'.
        """

        prompt_explicit = f"""
        You are a node in a 2D lattice system. Your current state is '{self.state}'.
        Your neighboring nodes have the following states: {neighbors_states}.
        Do you want to align with them or not? 
        Based on these conditions, you can only respond with either 'yes' or 'no' with equal prior.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=config['model'],
                messages=[{"role": "system", "content": prompt_explicit}],
                temperature=temperature,
                max_tokens=config['max_tokens']
            )
            new_state = response.choices[0].message.content.strip().lower()
            #new_state = new_state.encode('ascii', errors='ignore').decode('ascii')
            #raw_state = response.choices[0].message.content.strip()
            # Clean response to remove problematic characters before using/printing
            #new_state = raw_state.encode('ascii', errors='ignore').decode('ascii').lower()
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
        
        self.output_file = f"magnetizations/isingllm_magnetization__seed-{config['seed']}_model-{config['model']}_size-{self.size}_temp-{self.temperature}.csv"
        #self.output_file = f"magnetizations/isingllm_magnetization__seed-{config['seed']}_model-llama-4-scout-17b-16e-instruct_size-{self.size}_temp-{self.temperature}.csv"
        
        # Initialize magnetization file
        with open(self.output_file, 'w') as f:
            f.write("Step,Magnetization\n")
        
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
        # Save with step number in filename
        
        step_filename = f"lattices/lattice-state__step_{step}_seed-{config['seed']}_model-{config['model']}_size-{self.size}_temp-{self.temperature}.npy"
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
        filename += f"__seed-{config['seed']}_model-{config['model']}_size-{self.size}_temp-{self.temperature}.png"
        #filename += f"__seed-{config['seed']}_model-llama-4-scout-17b-16e-instruct_size-{self.size}_temp-{self.temperature}.png"

        
        # Save the figure
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close()
        
        #print(f"Lattice state image saved as {filename}")

    #def save_magnetization(self, filename: str):
    #    """Save magnetization history to a file."""
    #    with open(filename, 'w') as f:
    #        f.write("Step,Magnetization\n")  # CSV header
    #        for step, mag in enumerate(self.magnetization_history):
    #            f.write(f"{step},{mag:.4f}\n")

    def save_step_magnetization(self, step: int, magnetization: float):
        """Save magnetization for current step immediately to file."""
        with open(self.output_file, 'a') as f:
            f.write(f"{step},{magnetization:.4f}\n")