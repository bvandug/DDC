import matlab.engine
import neat
import dill
import neat.config
import numpy as np
import matplotlib.pyplot as plt
import os


def fitness(voltage_lst):
    '''
    Fitness function for the BBC.

    Args:
        voltage_lst (List[float]) : The array of observed output voltages.

    Returns:
        fitness (float): Calculated fitness score.
    '''
    dt = 5e-6
    t_max = 0.15
    n_max = t_max/dt
    fitness_sum = 0
    error = 10 #np.abs(goal/4)

    for k in range(len(voltage_lst)):
        if (goal-error) <= voltage_lst[k] <= (goal+error):
            fitness_sum += (error - abs(voltage_lst[k]-goal))**2 / error
        else:
            fitness_sum += 0

    fitness = fitness_sum/n_max
    
    print("Fitness score:", round(fitness,4))
    return fitness


def get_data():
    '''
    Collects and returns the workspace data.

    Returns:
        voltage_lst (List[float]) : Observed voltages.
        time_lst (List[float]) : Simulation times.
    '''
    # Get voltages
    voltage_2d = eng.eval("out.voltage")
    voltage_lst = []
    for voltage in voltage_2d:
        voltage_lst.append(voltage[0])

    # Get time
    time_2d = eng.eval("out.tout")
    time_lst = []
    for time in time_2d:
        time_lst.append(time[0]) 

    return voltage_lst, time_lst


def eval_genomes(genomes, config):
    '''
    Evaluates the fitness of each genome in the generation.

    Args:
        genomes (Any) : The list of genomes.
        config (Any) : The configuration of the population.
    '''
    global goal
    global final_goal
    final_goal = -30
    model = "bbcSimNEAT"
    fitness_lst = []

    for genome_id, genome in genomes:
        
        # Create and send ANN to controller
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        with open('network.dill', 'wb') as f:
            dill.dump(net, f)

        # Slightly vary goal and set goal
        goal = np.random.uniform(final_goal-2, high=(final_goal+2))
        goal_str = str(goal)       
        eng.set_param(f'{model}/Constant', 'Value', f'[{goal_str}]', nargout=0)

        # Set random noise
        noise_seed = str(np.random.randint(1, high=40000))
        eng.set_param(f'{model}/Noise', 'seed', f'[{noise_seed}]', nargout=0)
        noise_power = 0
        eng.set_param(f'{model}/Noise', 'Cov', f'[{noise_power}]', nargout=0)
        
        # Evaluate and assign fitness for genome
        eng.eval(f"out = sim('{model}.slx');", nargout=0)
        voltage_lst, time_lst = get_data()
        fitness_score = fitness(voltage_lst)
        genome.fitness = fitness_score
        fitness_lst.append(fitness_score)

    # Write fitness scores to file
    if os.path.exists('bbcFitness.dill'):
        with open('bbcfitness.dill', 'rb') as f:
            existing_lst = dill.load(f)
    else:
        existing_lst = []
    existing_lst.append(fitness_lst)
    with open('bbcFitness.dill', 'wb') as f:
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

    # Create the population and add stats reporter.
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    # Set up MATLAB engine
    print("Setting up engine...")
    global eng
    eng = matlab.engine.start_matlab()
    eng.load_system("bbcSimNEAT", nargout=0)

    if os.path.exists('bbcFitness.dill'):
        os.remove('bbcFitness.dill')

    # Run NEAT for 15 generations and return the best genome
    pop.run(eval_genomes, 15)
    return stats.best_genome()  


def show_winner(winner_lst):
    '''
    Plot results of the winning genome.

    Args:
        winner_lst (List[Any]) : Best genomes for each desired voltage. 
    '''
    print("Running winner ANNs...")
    model = 'bbcSimNEAT'
    goal_lst = [-30,-80,-110]
    s=0
    
    # Run winners for different desired voltages
    for winner in winner_lst:

        # Create winner ANN and send to controller
        net = neat.nn.FeedForwardNetwork.create(winner, config)
        with open('network.dill', 'wb') as f:
                dill.dump(net, f)
        
        # Set goal voltage
        final_goal = goal_lst[s]
        s+=1
        goal_str = str(final_goal)       
        eng.set_param(f'{model}/Constant', 'Value', f'[{goal_str}]', nargout=0)

        # Set random noise
        noise_seed = str(np.random.randint(1, high=40000))
        eng.set_param(f'{model}/Noise', 'seed', f'[{noise_seed}]', nargout=0)
        noise_power = 0
        eng.set_param(f'{model}/Noise', 'Cov', f'[{noise_power}]', nargout=0)

        # Run winner and get data
        eng.eval(f"out = sim('{model}.slx');", nargout=0)
        voltage_lst, time_lst = get_data()

        # Get stability time
        stab_time = None
        for j in range(len(voltage_lst)):
            if (final_goal+(0.04*final_goal)) <= voltage_lst[j] <= (final_goal-(0.04*final_goal)):
                if stab_time == None:
                    stab_time = time_lst[j]
            else:
                stab_time = None
        if stab_time == None:
            print("Failed to stabalise")
        else:
            print(f"Stabilisation time for {final_goal}V:", round(stab_time,3))

        # Plot results
        plt.plot(time_lst, voltage_lst, label = f'Goal Voltage: {final_goal}V')

        if s == 3:
            plt.axhline(y=(final_goal+(0.04*final_goal)), linestyle='--', color = 'k', label='$\pm$ 4% error bar')
        else:
            plt.axhline(y=(final_goal+(0.04*final_goal)), linestyle='--', color = 'k')
        plt.axhline(y=(final_goal-(0.04*final_goal)), linestyle='--', color = 'k')

    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.xlim(0,0.07)
    plt.ylim(-120, 8)
    plt.legend()
    plt.savefig('plot_test.pdf', format='pdf')
    plt.show()

    print("Complete.")


def main():
    '''
    Run NEAT, get the winner, and plot.
    '''
    winner = run("bbcConfig.txt")
    with open('bbcWinner.dill', 'wb') as f:
            dill.dump(winner, f)
    show_winner([winner])
    eng.quit()

if __name__ == '__main__':
    main()


# def plot_winner():
#     '''
#     Uncomment this method, and comment out main method to run the winners again.
#     '''
#     with open('bbcWinner30V5.dill', 'rb') as f:
#         winner30V = dill.load(f)

#     with open('bbcWinner80V4.dill', 'rb') as f:
#         winner80V = dill.load(f)

#     with open('bbcWinner110V3.dill', 'rb') as f:
#         winner110V = dill.load(f)
    
#     global config
#     config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
#                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
#                                 "bbcConfig.txt")

#     print("Setting up engine...")
#     global eng
#     eng = matlab.engine.start_matlab()
#     eng.load_system('bbcSimNEAT', nargout=0)
#     show_winner([winner30V, winner80V, winner110V])
#     eng.quit()
# plot_winner()