import yaml
from ising_llms import IsingLLM
import time
from tqdm import tqdm
import numpy as np
import os



if __name__ == "__main__":

    # Load the config file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

            # Create directories if they don't exist
    os.makedirs('magnetizations', exist_ok=True)
    os.makedirs('images', exist_ok=True)
    os.makedirs('lattices', exist_ok=True)


    print(f"Running {config['model']} on lattice {config['size']}^2 with T = {config['temperature']} and prompt = {config['prompt_type']}")
    # Create a size x size lattice
    model = IsingLLM(size=config['size'], temperature=config['temperature'])
    
    # Run simulation for a few steps
    print("Init step")
    if config['show_viz']:
        model.visualize()
    
    # Calculate and store initial magnetization
    initial_magnetization = model.calculate_magnetization()
    #model.magnetization_history.append(initial_magnetization)
    model.save_step_magnetization(0, initial_magnetization)
    print(f"Initial Magnetization: {initial_magnetization:.2f}\n")
    
    # Save initial state
    model.save_lattice_state(0)
    
    for step in tqdm(range(config['num_steps'])):
        #print(f"Step {step}")
        model.step()
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
            print(f"Step: {step}, Magnetization: {magnetization:.2f}\n")
        time.sleep(1)  # Delay to respect API rate limits
    
    # Save magnetization history to file
    #output_file = f"results/isingllm_magnetization__model-{config['model']}_size-{config['size']}_temp-{config['temperature']}.csv"
    #model.save_magnetization(output_file)
    #print(f"Magnetization history saved to {output_file}")