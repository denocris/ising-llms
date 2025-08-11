import yaml
import argparse
from ising_llms import IsingLLM
import time
from tqdm import tqdm
import numpy as np
import os
import random
import torch

from openai import OpenAI
from groq import Groq
from transformers import AutoModelForCausalLM, AutoTokenizer


if __name__ == "__main__":

    # Load the config file
    #with open('config.yaml', 'r') as file:
    #    config = yaml.safe_load(file)

    def load_config(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to your config YAML (e.g. config-gpt.yaml)."
    )
    args = parser.parse_args()
    config = load_config(args.config)
    print("Running with config:", args.config)
    
    # Create directories if they don't exist
    os.makedirs('magnetizations', exist_ok=True)
    os.makedirs('images', exist_ok=True)
    os.makedirs('lattices', exist_ok=True)

    #random.seed(config['seed'])    
    #np.random.seed(config['seed']) 
    #torch.manual_seed(config['seed']) 

    if config['service']=='groq':
        model_name = config['model']
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        client = {"model": model_name, "client": Groq(api_key=GROQ_API_KEY), "service": "groq"}
    elif config['service']=='openai':
        model_name = config['model']
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        client = {"model": model_name, "client": OpenAI(api_key=OPENAI_API_KEY), "service": "openai"}
    elif config['service']=='huggingface':
        model_name = config['model']
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        client = {"model": model_name, "tokenizer": tokenizer, "service": "huggingface"}
    elif config['service']=='ollama':
        model_name = config['model']
        client = {"model": model_name, "client": None, "service": "ollama"}
    else:
        print('Select a client!')

    #for current_seed in range(config['seed'], config['seed_range']):
    for current_seed in config['seed_range']:
        # update the seed 
        random.seed(current_seed)    
        np.random.seed(current_seed) 
        torch.manual_seed(current_seed) 

        for temp in config['temperature']:
            ############ INITIALIZE SEED #############
            # inject the current temperature
            #config['temperature'] = temp

            print(f"Model: {config['model']}, Seed: {current_seed}, Lattice: {config['size']}^2, T = {temp}, prompt = {config['prompt_type']}, Z2 = {config['z2_symmetry']}")
            # Create a size x size lattice
            model = IsingLLM(size=config['size'], 
                             temperature=temp,
                             client=client,
                             random_seed=current_seed,
                             prompt_type=config['prompt_type'],
                             max_tokens=config['max_tokens'],
                             z2_symmetry=config['z2_symmetry'])
            
            # Run simulation for a few steps
            print("Init step")
            if config['show_viz']:
                model.visualize()
            
            # Calculate and store initial magnetization
            initial_magnetization = model.calculate_magnetization()
            #model.magnetization_history.append(initial_magnetization)
            model.save_step_magnetization(0, initial_magnetization)
            print(f"Initial Magnetization: {initial_magnetization:.8f}\n")
            
            # Save initial state
            model.save_lattice_state(0)
            
            for step in tqdm(range(config['num_steps'])):
                #print(f"Step {step}")
                model.step(step)
                if config['show_viz']:
                    print(f"Step {step + 1}")
                    model.visualize()
                if config['generate_image']:
                    model.save_lattice_image(step=step+1)  # Save image for this step
                model.save_lattice_state(step + 1)

                # Save magnetization and lattice state
                magnetization = model.calculate_magnetization()
                model.save_step_magnetization(step + 1, magnetization)
                
                if step%5==0:
                    print(f"Step: {step}, Magnetization: {magnetization:.8f}\n")
                time.sleep(1)  # Delay to respect API rate limits
        
