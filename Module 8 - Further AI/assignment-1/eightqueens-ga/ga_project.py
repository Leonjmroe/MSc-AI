# %% [markdown]
# ### Genetic Algorithms - 8 Queen Puzzle Assignment
# -----
# 

# %% [markdown]
# #### Task 1 - Standard Implementation

# %%
# Imports

import pandas as pd
import numpy as np
import time
import plotly.express as px
import random


# %%
# Queens State

class QueensState:
    def __init__(self, board_size, state=None):
        if state is None:
            # Inital board set up
            self.state = np.random.randint(0, board_size, board_size)
        else:
            self.state = np.array(state) 

        # Set the goal fitness value where there is no queen attacks 
        self.goal_fitness = int(((board_size ** 2) - board_size) / 2)
    
    
    def fitness(self):
        count = 0
        n = len(self.state)
        
        for i in range(n - 1):
            remaining_queens = self.state[i + 1:]
            current_queen = self.state[i]
            
            # Count of the queen attacks on the row 
            count += (current_queen == remaining_queens).sum()
            
            # Count of the queen attacks on the diagonal 
            distances = np.arange(1, n - i)
            upper_diagonal = current_queen + distances
            lower_diagonal = current_queen - distances
            
            # Total queen on queen attacks
            count += sum((remaining_queens == upper_diagonal) | (remaining_queens == lower_diagonal))

        # Return the final fitness 
        return self.goal_fitness - count

# %%
# Standard 8 Queen Genetic Algorithm

class Standard8QueenGA:
    def __init__(self, population_size, mutation_probability, initial_random_state):
        self.board_size = 8
        self.population_size = population_size
        self.mutation_probability = mutation_probability
        self.last_state = None

        # Initial population - unique neighbors of a random initial state
        initial_state = np.random.randint(0, self.board_size, self.board_size)
        self.current_population = [QueensState(self.board_size, self.random_neighbour(initial_state)) for _ in range(population_size)]


    def random_neighbour(self, state):
        new_state = state.copy() 

        # Choose a random column
        column = np.random.choice(range(self.board_size)) 

        # Create a list of possible row positions excluding the current row of the queen
        possible_rows = [row for row in range(self.board_size) if row != state[column]]

        # Set the queen in the chosen column to a new row position
        new_state[column] = np.random.choice(possible_rows)

        return new_state


    def calculate_population_fitness(self):
        # Calculate fitness and selection probability for each individual in the population
        population_data = []
        total_fitness = 0
        
        for individual in self.current_population:
            fitness = individual.fitness()
            population_data.append({
                'state': individual.state,
                'fitness': fitness
            })
            total_fitness += fitness
        
        # Calculate selection probability for each individual based on fitness
        for data in population_data:
            data['selection_prob'] = data['fitness'] / total_fitness
            
        # Return the sorted the population by the highest selection probabilities
        return sorted(population_data, key=lambda x: x['selection_prob'], reverse=True)


    def select_parent(self, population_data):
        # Generate a random number between 0 and 1
        random_value = np.random.random()
        
        # Use cumulative probability to select an individual
        cumulative_probability = 0.0
        for individual in population_data:
            cumulative_probability += individual['selection_prob']
            if cumulative_probability > random_value:
                return individual['state']


    def crossover(self, parent1, parent2):
        # Perform a single-point crossover to produce two offspring via randomisation
        idx = np.random.randint(1, len(parent1))  

        # Apply the cross over of two parents to product two children 
        return np.concatenate([parent1[:idx], parent2[idx:]]), np.concatenate([parent2[:idx], parent1[idx:]])


    def mutate(self, state):
        # Randomly mutate a queen's position in one column with a certain probability
        if np.random.random() < self.mutation_probability:

             # Select a random column to mutate
            idx = np.random.randint(len(state))  

            # Randomly choose a new row position for the queen in the selected column, excluding current row 
            state[idx] = np.random.choice(list(set(range(self.board_size)) - {state[idx]}))
        return state


    def create_next_generation(self, population_data):
        # Generate the next generation by selecting parents, applying crossover, and mutating offspring
        next_generation = []
        
        # Fill the rest of the population with offspring
        while len(next_generation) < self.population_size:
            parent1 = self.select_parent(population_data)
            parent2 = self.select_parent(population_data)
            
            # Create two children through crossover
            child1, child2 = self.crossover(parent1, parent2)
            
            # Apply mutation to each child
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            # Add children to the next generation
            next_generation.extend([
                QueensState(self.board_size, child1),
                QueensState(self.board_size, child2)
            ])
        
        # Ensure the population size remains constant
        self.current_population = next_generation[:self.population_size]


    def run(self):
        generation = 0
        fitness_history = []
        
        while generation < 5000:
            population_data = self.calculate_population_fitness()
            
            # Check if the best individual in the population is a solution (28 for a standard chess board)
            if population_data[0]['fitness'] == 28:

                self.last_state = population_data[0]['state']
                # print(f"Solution found in generation {generation}")
                return fitness_history, generation  # Return fitness history and generation where solution is found
            
            # Track the best fitness for each generation (optional for progress analysis)
            best_fitness = population_data[0]['fitness']
            fitness_history.append(best_fitness)
            
            # Generate the next generation
            self.create_next_generation(population_data)
            
            generation += 1

        return fitness_history, generation 
    

    def display_solution(self):
        # Display the chessboard with queens in the positions defined by self.last_state
        for row in range(self.board_size):
            print(f'{self.board_size - row} |', end=' ')
            for col in range(self.board_size):
                if self.last_state[col] == row:
                    print('Q ', end='')  
                else:
                    print('- ', end='') 
            print(f'|')

# %%
# Run Standard 8 Queen Genetic Algorithm (10 trial example)

def run_8_queen_ga_solver(trails):

    for trial in range(trails):

        start_time = time.time()
        
        ga = Standard8QueenGA(
            population_size=300,
            mutation_probability=0.1,
            initial_random_state=np.random.randint(0, 8, 8)
        )
        
        ga.run()

        print(f'Solution {trial + 1}:')

        ga.display_solution()

        print(f'Run Time: {(time.time() - start_time):.2f} seconds\n')


# Run the standard 8 queen GA solver for 10 random trials
run_8_queen_ga_solver(10)

# %% [markdown]
# #### Task 2 - Extension and Optimisation

# %%
# Standard8QueenGA Hyperparameter Tuning 

def optimise_parameters(trials=10):

    # Parameter ranges
    population_sizes = [20, 50, 100, 250, 500]
    mutation_probabilities = [0.01, 0.03, 0.8, 0.15, 0.3]

    best_parameters = None
    best_average_generations = float('inf')

    # Test each combination
    for population_size in population_sizes:
        for mutation_probability in mutation_probabilities:
            generations_list = []
            times_list = []
            
            # Run multiple trials for each combination 
            for _ in range(trials):

                start_time = time.time()
    
                ga = Standard8QueenGA(
                    population_size=population_size,
                    mutation_probability=mutation_probability,
                    initial_random_state=np.random.randint(0, 8, 8)
                )
            
                _, generations = ga.run()
          
                generations_list.append(generations)
                times_list.append(time.time() - start_time)
            
            # Calculate averages
            average_generations = np.mean(generations_list)
            average_time = np.mean(times_list)
            max_time = max(times_list)
            
            # Update best if better param found
            if average_generations < best_average_generations:
                best_avg_generations = average_generations
                best_parameters = (population_size, population_sizes)

            print(f'Population Size: {population_size}, Mutation Probability: {mutation_probability:.2f}')
            print(f'Average Generations: {average_generations:.1f}')
            print(f'Average Time: {average_time:.2f}s\n')
    
    # Print final parameters 
    print('\nBest Parameters:')
    print(f"Population Size: {best_parameters[0]}")
    print(f"Mutation Probability: {best_parameters[1]}")

    return best_parameters


# Run optimisation
best_params = optimise_parameters()

# %%
# Horses State 

class HorsesState:
    def __init__(self, board_size, horse_count, state=None):
        self.board_size = board_size
        self.horse_count = horse_count
        
        if state is None:
            # Create permutation of positions to ensure no overlap
            positions = []
            for i in range(horse_count):
                while True:
                    x = np.random.randint(0, board_size)
                    y = np.random.randint(0, board_size)
                    if [x,y] not in positions:  # Simple check for uniqueness
                        positions.append([x,y])
                        break
            self.state = np.array(positions)
        else:
            self.state = np.array(state)
    
    def fitness(self):
        count = 0
        
        # Check each horse against others
        for i, horse1 in enumerate(self.state[:-1]):
            for horse2 in self.state[i+1:]:
    
                x = abs(horse1[0] - horse2[0])
                y = abs(horse1[1] - horse2[1])
                
                # If horses can attack each other 
                if (x == 2 and y == 1) or (x == 1 and y == 2):
                    count += 1
                    
        # Maximum possible attacks 
        max_attacks =  self.horse_count * ( self.horse_count - 1) // 2
        return max_attacks - count


# %%
import numpy as np

class ChainQueensState:
    """
    A variant of N-Queens where some queens must form a chain via knight moves.
    A valid solution requires:
    1. No queen attacks (standard N-Queens rules)
    2. At least 3 queens connected by knight's moves
    """
    def __init__(self, board_size, state=None):
        self.board_size = board_size
        # Initialize random queen positions or use provided state
        self.state = np.random.randint(0, board_size, board_size) if state is None else np.array(state)
        
        # Goal fitness includes both queen conflicts and chain requirement
        n = board_size
        self.goal_fitness = (n * (n - 1)) // 2 + n  # Standard queen conflicts + chain penalty
    
    def is_knight_move(self, row1, col1, row2, col2):
        """Check if two positions form a valid knight's move pattern"""
        row_diff = abs(row1 - row2)
        col_diff = abs(col1 - col2)
        # Knight moves in L-shape: 2 squares one way, 1 square perpendicular
        return (row_diff == 2 and col_diff == 1) or (row_diff == 1 and col_diff == 2)
    
    def has_three_chain(self):
        """Check if board has at least 3 queens connected by knight moves"""
        # Look for any chain of 3 queens
        for i in range(len(self.state)):
            for j in range(i + 1, len(self.state)):
                # Check first potential link
                if self.is_knight_move(self.state[i], i, self.state[j], j):
                    # If found, look for third queen to complete chain
                    for k in range(j + 1, len(self.state)):
                        if self.is_knight_move(self.state[j], j, self.state[k], k):
                            return True
        return False
    
    def fitness(self):
        """
        Calculate fitness score based on:
        1. Number of queen attacks (lower is better)
        2. Presence of required chain (big penalty if missing)
        """
        conflicts = 0
        n = len(self.state)
        
        # Count standard queen conflicts (row and diagonal)
        for i in range(n-1):
            for j in range(i+1, n):
                if (self.state[i] == self.state[j] or  # Same row
                    abs(self.state[i] - self.state[j]) == abs(i - j)):  # Diagonal
                    conflicts += 1
        
        # Add penalty if missing required chain of 3 queens
        if not self.has_three_chain():
            conflicts += n
            
        return self.goal_fitness - conflicts

# %%
# Optimised and Extended Genetic Algorithm

class OptimisedExtendedGA:
    def __init__(self, board_size, population_size, mutation_probability, max_generations, mutation_type, piece_type, horse_count=None):
        self.board_size = board_size
        self.population_size = population_size
        self.mutation_probability = mutation_probability
        self.max_generations = max_generations
        self.mutation_type = mutation_type
        self.piece_type = piece_type
        self.horse_count = horse_count
        self.last_state = None
        self.fitness_history = []

        if piece_type == 'Queen':
            # ***** Optimisation 1 *****
            self.current_population = [QueensState(board_size) for _ in range(population_size)]
            # Dynamic goal fitness for different board sizes
            self.goal_fitness = int(((board_size ** 2) - board_size) / 2)

        # ***** Extension 2 *****
        if piece_type == 'Horse':
            self.current_population = [HorsesState(board_size, horse_count) for _ in range(population_size)]
            self.goal_fitness = (horse_count * (horse_count - 1)) // 2

        # ***** Extension 3 *****
        if piece_type == 'ChainQueen':
            self.current_population = [ChainQueensState(board_size) for _ in range(population_size)]
            # Match the ChainQueensState goal fitness exactly
            queen_conflicts = (board_size * (board_size - 1)) // 2
            chain_requirement = board_size
            self.goal_fitness = queen_conflicts + chain_requirement




    def calculate_population_fitness(self):
        population_data = []
        total_fitness = 0
        for individual in self.current_population:
            fitness = individual.fitness()
            population_data.append({
                'state': individual.state,  
                'fitness': fitness
            })
            total_fitness += fitness

        for data in population_data:
            data['selection_prob'] = data['fitness'] / total_fitness
        return sorted(population_data, key=lambda x: x['selection_prob'], reverse=True)



    def select_parent(self, population_data):
        random_value = np.random.random()
        cumulative_probability = 0.0
        for individual in population_data:
            cumulative_probability += individual['selection_prob']
            if cumulative_probability > random_value:
                return individual['state']


    # ***** Optimisation 2 *****
    def crossover(self, parent1, parent2):
        # Apply the cross over fixed at 50% of each  parent 
        return np.concatenate([parent1[:4], parent2[4:]]), np.concatenate([parent2[:4], parent1[4:]])


    # ***** Extension 1 *****
    def smart_mutate(self, state):
        if np.random.random() >= self.mutation_probability:
            return state
            
        # Count attacks for each queen
        conflicts = []
        for i, row in enumerate(state):
            conflict_count = 0
            for j, other_row in enumerate(state):
                # Skip comparing queen with itself
                if i == j:
                    continue
                # Check same row or diagonal attacks
                if (row == other_row or abs(row - other_row) == abs(i - j)):
                    conflict_count += 1
            conflicts.append(conflict_count)
        
        # Find the queen with the most attacks
        worst_queen = np.argmax(conflicts)
        current_conflicts = conflicts[worst_queen]
        
        # Try each row position for worst queen
        best_position = state[worst_queen]  
        least_conflicts = current_conflicts
        
        for new_row in range(self.board_size):
            if new_row == state[worst_queen]:
                continue
                
            # Count attacks in this new position
            conflicts = 0
            for col, row in enumerate(state):
                if col == worst_queen:
                    continue
                if (new_row == row or 
                    abs(new_row - row) == abs(worst_queen - col)):
                    conflicts += 1
                    
            if conflicts < least_conflicts:
                least_conflicts = conflicts
                best_position = new_row
        
        # Move the queen to best spot found
        state[worst_queen] = best_position
        return state

    
    def simple_mutate(self, state):
        if np.random.random() < self.mutation_probability:
            idx = np.random.randint(len(state))  
            state[idx] = np.random.choice(list(set(range(self.board_size)) - {state[idx]}))
        return state
    
     # ***** Extension 3 *****
    def mirror_mutate(self, state):
        """Only mutate real queens, reflections follow automatically"""
        if np.random.random() < self.mutation_probability:
            idx = np.random.randint(len(state))
            state[idx] = np.random.choice(
                list(set(range(self.board_size)) - {state[idx]})
            )
        return state
    

    # ***** Extension 2 *****
    def simple_horse_mutate(self, state):
        if np.random.random() < self.mutation_probability:
            horse_idx = np.random.randint(self.horse_count)
            existing_positions = [list(pos) for pos in state]
            while True:
                new_x = np.random.randint(self.board_size)
                new_y = np.random.randint(self.board_size)
                if [new_x, new_y] not in existing_positions:
                    state[horse_idx] = [new_x, new_y]
                    break
        return state

    # *** unclacified *** 
    def fitness_stagnation(self, lookback):
        if len(self.fitness_history) > lookback:
            recent_fitness = self.fitness_history[-lookback:]
            return recent_fitness[-1] > recent_fitness[0]
        return False


    def create_next_generation(self, population_data):
        next_generation = []
        while len(next_generation) < self.population_size:
            parent1 = self.select_parent(population_data)
            parent2 = self.select_parent(population_data)
            child1, child2 = self.crossover(parent1, parent2)

            if self.mutation_type == 'Simple':

                if self.piece_type == 'Queen':
                    child1 = self.simple_mutate(child1)
                    child2 = self.simple_mutate(child2)  

                # ***** Extention 2 *****
                if self.piece_type == 'Horse':
                    child1 = self.simple_horse_mutate(child1)
                    child2 = self.simple_horse_mutate(child2) 

                # ***** Extention 2 *****
                if self.piece_type == 'MirroredQueen':
                    child1 = self.mirror_mutate(child1)
                    child2 = self.mirror_mutate(child2)   

            # ***** Extension 1 *****
            if self.mutation_type == 'Smart':
                child1 = self.smart_mutate(child1)
                child2 = self.smart_mutate(child2)

            # *** unclacified *** 
            # if self.fitness_stagnation(100):
            #     print('stuck for 100 gens')
            #     child1 = self.mutate(child1)
            #     child2 = self.mutate(child2)
        
            if self.piece_type == 'Queen':
                next_generation.extend([
                    QueensState(self.board_size, child1),
                    QueensState(self.board_size, child2)
                ])

            # ***** Extension 2 *****
            if self.piece_type == 'Horse':
                next_generation.extend([
                    HorsesState(self.board_size, self.horse_count, child1),
                    HorsesState(self.board_size, self.horse_count, child2)
                ])

            # ***** Extension 3 *****
            if self.piece_type == 'ChainQueen':
                next_generation.extend([
                    ChainQueensState(self.board_size, child1),
                    ChainQueensState(self.board_size, child2)
                ])

        self.current_population = next_generation[:self.population_size]




    def run(self):
        generation = 0
        while generation < self.max_generations:
            population_data = self.calculate_population_fitness()
            if population_data[0]['fitness'] == self.goal_fitness:
                self.last_state = population_data[0]['state']
                return self.fitness_history, generation, True 
            best_fitness = population_data[0]['fitness']
            self.fitness_history.append(best_fitness)
            self.create_next_generation(population_data)
            generation += 1
        return self.fitness_history, generation, False 
    

    def display_solution(self):

        if self.piece_type == 'Queen':
            for row in range(self.board_size):
                if (self.board_size - row) < 10:
                    print(f'0{self.board_size - row} |', end=' ')
                else:
                    print(f'{self.board_size - row} |', end=' ')
                for col in range(self.board_size):
                    if self.last_state[col] == row:
                        print('Q ', end='')
                    else:
                        print('- ', end='')
                print('|')

        # ***** Extension 2 *****
        if self.piece_type == 'Horse':
            board = [[' - ' for _ in range(self.board_size)] for _ in range(self.board_size)]
            
            # Place horses using coordinates
            for horse_pos in self.last_state:
                x, y = horse_pos
                board[x][y] = ' H '
            
            for row in range(self.board_size):
                if (self.board_size - row) < 10:
                    print(f'0{self.board_size - row} |', end=' ')
                else:
                    print(f'{self.board_size - row} |', end=' ')
                for col in range(self.board_size):
                    print(f'{board[row][col]} ', end='')
                print('|')

        # ***** Extension 3 *****
        if self.piece_type == 'ChainQueen':
            """Display chain queens solution with chain indicators"""
            # Create empty board
            board = [['-' for _ in range(self.board_size)] for _ in range(self.board_size)]
            
            # Place all queens
            for col, row in enumerate(self.last_state):
                board[row][col] = 'Q'
            
            # Mark queens that form knight-move chains
            for i in range(len(self.last_state)):
                for j in range(i + 1, len(self.last_state)):
                    # Check if queens form knight's move
                    row1, col1 = self.last_state[i], i
                    row2, col2 = self.last_state[j], j
                    row_diff = abs(row1 - row2)
                    col_diff = abs(col1 - col2)
                    if (row_diff == 2 and col_diff == 1) or (row_diff == 1 and col_diff == 2):
                        board[row1][col1] = 'Q*'
                        board[row2][col2] = 'Q*'
            
            # Print board
            for row in range(self.board_size):
                if (self.board_size - row) < 10:
                    print(f'0{self.board_size - row} |', end=' ')
                else:
                    print(f'{self.board_size - row} |', end=' ')
                for col in range(self.board_size):
                    print(f'{board[row][col]} ', end='')
                print('|')
            print("\nNote: Q* indicates queens that form knight-move chains")

# %%
# Extension 1 - The Crowded Queens Puzzle 

def run_crowded_queens_puzzle(trails):

    for trial in range(trails):

        start_time = time.time()
        
        ga = OptimisedExtendedGA(
            board_size=20,
            population_size=250,
            mutation_probability=1,
            max_generations=250,
            mutation_type='Smart',
            piece_type='Queen'
        )
        
        _, generation, solution = ga.run()

        if solution:
            print(f'Solution {trial + 1}:')
            ga.display_solution()
            print(f'Run Time: {(time.time() - start_time):.2f}s, Generatios: {generation}\n')
        else:
            print(f'No solution found in {generation} generations\n')


run_crowded_queens_puzzle(10)

# %%
# Extension 2 - The Cavalry Puzzle 

def run_cavalry_puzzle(trails):

    for trial in range(trails):

        start_time = time.time()
        
        ga = OptimisedExtendedGA(
            board_size=20,
            population_size=400,
            mutation_probability=1,
            max_generations=1000,
            mutation_type='Simple',
            piece_type='Horse',
            horse_count=35
        )
        
        _, generation, solution = ga.run()

        if solution:
            print(f'Solution {trial + 1}:')
            ga.display_solution()
            print(f'Run Time: {(time.time() - start_time):.2f}s, Generatios: {generation}\n')
        else:
            print(f'No solution found in {generation} generations\n')


run_cavalry_puzzle(1)

# %%
# Extension 3 - The Chained Queens Puzzle 

def run_chained_queens_puzzle(trails):

    for trial in range(trails):

        start_time = time.time()
        
        ga = OptimisedExtendedGA(
            board_size=12,
            population_size=500,
            mutation_probability=0.5,
            max_generations=1000,
            mutation_type='Smart',
            piece_type='ChainQueen'
        )
        
        _, generation, solution = ga.run()

        if solution:
            print(f'Solution {trial + 1}:')
            ga.display_solution()
            print(f'Run Time: {(time.time() - start_time):.2f}s, Generatios: {generation}\n')
        else:
            print(f'No solution found in {generation} generations\n')


run_chained_queens_puzzle(1)

# %%



