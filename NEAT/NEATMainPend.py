import matlab.engine
import neat
import dill
import neat.config
import numpy as np
import matplotlib.pyplot as plt
import NEATactivationsPend as activations
import os
import time


def fitness(angle_lst):
    '''
    Fitness function for the inverted pendulum.

    Args:
        angle_lst (List[float]) : Observed angles.
    
    Returns:
        fitness (float): Calculated fitness score.
    '''
    dt = 0.001
    t_max = 4
    n_max = t_max/dt
    fitness_sum = 0

    for k in range(len(angle_lst)):
        if -np.pi/2 <= angle_lst[k] <= np.pi/2:
            fitness_sum += (1-np.abs(2*angle_lst[k]/np.pi))**2

    fitness = (fitness_sum)/n_max
    
    print("Fitness score:", round(fitness,4))
    return fitness


def get_data():
    '''
    Collects and returns the workspace data.

    Returns:
        angle_lst (List[float]) : Observed angles.
        time_lst (List[float]) : Simulation times.
    '''
    # Get angles
    angle_2d = eng.eval("out.angle")
    angle_lst = []
    for angle in angle_2d:
        angle_lst.append(angle[0])

    # Get time
    time_2d = eng.eval("out.tout")
    time_lst = []
    for time in time_2d:
        time_lst.append(time[0]) 
        
    return angle_lst, time_lst


def eval_genomes(genomes, config):
    '''
    Evaluates the fitness of each genome in the generation.

    Args:
        genomes (Any) : The list of genomes.
        config (Any) : The configuration of the population.
    '''
    model = 'pendSimNEAT'
    fitness_lst = []

    for genome_id, genome in genomes:
        
        # Create and send ANN to controller
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        with open('network.dill', 'wb') as f:
            dill.dump(net, f)

        # Set random intial angle
        intial_angle = np.random.uniform(-1, 1)
        while intial_angle <= 0.05 and intial_angle >= -0.05:
            intial_angle = np.random.uniform(-1, 1)
        eng.set_param(f'{model}/Pendulum and Cart', 'init', str(intial_angle), nargout=0)

        # Set random noise
        noise_seed = str(np.random.randint(1, high=40000))
        eng.set_param(f'{model}/Noise', 'seed', f'[{noise_seed}]', nargout=0)
        noise_power = 0
        eng.set_param(f'{model}/Noise', 'Cov', f'[{noise_power}]', nargout=0)
        noise_seed_v = str(np.random.randint(1, high=40000))
        eng.set_param(f'{model}/Noise_v', 'seed', f'[{noise_seed_v}]', nargout=0)
        noise_power_v = 0
        eng.set_param(f'{model}/Noise_v', 'Cov', f'[{noise_power_v}]', nargout=0)

        # Evaluate and assign fitness for genome
        eng.eval(f"out = sim('{model}.slx');", nargout=0)
        angle_lst, time_lst = get_data()
        fitness_score = fitness(angle_lst)
        genome.fitness = fitness_score
        fitness_lst.append(fitness_score)

    # Write fitness scores to file
    if os.path.exists('pendFitness.dill'):
        with open('pendFitness.dill', 'rb') as f:
            existing_lst = dill.load(f)
    else:
        existing_lst = []
    existing_lst.append(fitness_lst)
    with open('pendFitness.dill', 'wb') as f:
        dill.dump(existing_lst, f)


def run(config_file):
    '''
    Configures the population and runs NEAT.

    Args:
        config_file (str) : The name of the txt configuration file.

    Returns:
        best_genome (Any) : The highest-performing genome. 
    '''
    # Load configuration
    global config
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)
    
    # Add my own activation functions
    activation_functions = activations.get_functions()
    for name, function in activation_functions:
        config.genome_config.add_activation(name, function)
    
    # Create the population and add stats reporter.
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    # Set up MATLAB engine
    print("Setting up engine...")
    global eng
    eng = matlab.engine.start_matlab()
    eng.load_system('pendSimNEAT', nargout=0)

    if os.path.exists('pendFitness.dill'):
        os.remove('pendFitness.dill')

    # Run NEAT for 25 generations and return the best genome
    pop.run(eval_genomes, 25)
    return stats.best_genome()   


def show_winner(winner):
    '''
    Plot results of the winning genome.

    Args:
        winner (Any) : Best genome.
    '''
    print("Running winner ANN...")
    model = 'pendSimNEAT'
    i = 0.2

    # Run winner for different initial angles
    while i <= 1:

        # Create winner ANN and send to controller
        net = neat.nn.FeedForwardNetwork.create(winner, config)
        with open('network.dill', 'wb') as f:
            dill.dump(net, f)
        
        # Set random intial angle
        initial_angle = round(i,3)
        i += 0.4
        eng.set_param(f'{model}/Pendulum and Cart', 'init', str(initial_angle), nargout=0)

        # Set random noise
        noise_seed = str(np.random.randint(1, high=40000))
        eng.set_param(f'{model}/Noise', 'seed', f'[{noise_seed}]', nargout=0)
        noise_power = 0
        eng.set_param(f'{model}/Noise', 'Cov', f'[{noise_power}]', nargout=0)
        noise_seed_v = str(np.random.randint(1, high=40000))
        eng.set_param(f'{model}/Noise_v', 'seed', f'[{noise_seed_v}]', nargout=0)
        noise_power_v = 0
        eng.set_param(f'{model}/Noise_v', 'Cov', f'[{noise_power_v}]', nargout=0)
        
        # Run winner and get data
        eng.eval(f"out = sim('{model}.slx');", nargout=0)
        angle_lst, time_lst = get_data()

        # Get stability time
        stab_time = None
        for j in range(len(angle_lst)):
            if -0.05 <= angle_lst[j] <= 0.05:
                if stab_time == None:
                    stab_time = time_lst[j]
            else:
                stab_time = None
        if stab_time == None:
            print("Failed to stabalise")
        else:
            print(f"Stabilisation time for {initial_angle} init. angle:", round(stab_time,3))

        # Plot results
        plt.plot(time_lst, angle_lst, label = f'{initial_angle} init. angle')

    plt.axhline(y=0.05, linestyle='--', color = 'k', label='$\pm$ 0.05 error bound')
    plt.axhline(y=-0.05, linestyle='--', color = 'k')
    plt.xlabel("Time (s)")
    plt.ylabel("Theta (rad)")
    plt.xlim(0,0.3)
    plt.ylim(-0.2,1)
    plt.legend()
    plt.show()

    print("Complete.")


def main():
    start_time = time.time()
    '''
    Run NEAT, get the winner, and plot.
    '''
    winner = run("pendConfig.txt")
    with open('pendWinner.dill', 'wb') as f:
            dill.dump(winner, f)
    show_winner(winner)
    duration = time.time() - start_time
    print(f"Simulation complete in {duration:.1f} secs")
    eng.quit()

if __name__ == '__main__':
    main()


# def plot_winner():
#     '''
#     Uncomment this method, and comment out main method to run the winner again.
#     '''
#     with open('pendWinner7.dill', 'rb') as f:
#         winner = dill.load(f)
    
#     global config
#     config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
#                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
#                                 "pendConfig.txt")
    
#     activation_functions = activations.get_functions()
#     for name, function in activation_functions:
#         config.genome_config.add_activation(name, function)

#     print("Setting up engine...")
#     global eng
#     eng = matlab.engine.start_matlab()
#     eng.load_system('pendSimNEAT', nargout=0)
#     show_winner(winner)
#     eng.quit()
# plot_winner()