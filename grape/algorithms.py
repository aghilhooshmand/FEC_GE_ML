# ===== MODIFICATIONS BY AGHIL FOR UGE INTEGRATION =====
# This file contains modifications made by Aghil for UGE (Unified Grammatical Evolution) integration:
# 1. Invalid individuals tracking (min/max/avg/std) - tracks number of invalid individuals per generation
# 2. Nodes length tracking (min/max/avg/std) - tracks number of terminal symbols per generation
# All modifications are clearly marked with "MODIFICATION BY AGHIL" comments
# ===== END MODIFICATIONS BY AGHIL =====
#
#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import random
import math
import numpy as np
import time
import warnings

from deap import tools

# Import cache tracking for FEC analysis
try:
    from uge.utils.helpers import set_cache_context
except ImportError:
    # Fallback if import fails
    def set_cache_context(run_id, generation):
        pass

def varAnd(population, toolbox, cxpb, mutpb,
           bnf_grammar, codon_size, max_tree_depth, codon_consumption,
           genome_representation, max_genome_length):
    """Part of an evolutionary algorithm applying only the variation part
    (crossover **and** mutation). The modified individuals have their
    fitness invalidated. The individuals are cloned so returned population is
    independent of the input population.

    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: A list of varied individuals that are independent of their
              parents.

    """
    offspring = [toolbox.clone(ind) for ind in population]

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i],
                                                          bnf_grammar, 
                                                          max_tree_depth, 
                                                          codon_consumption,
                                                          genome_representation,
                                                          max_genome_length)

    for i in range(len(offspring)):
        offspring[i], = toolbox.mutate(offspring[i], mutpb,
                                       codon_size, bnf_grammar, 
                                       max_tree_depth, codon_consumption,
                                       max_genome_length)

    return offspring

class hofWarning(UserWarning):
    pass
    
def ge_eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, elite_size, 
                bnf_grammar, codon_size, max_tree_depth, 
                max_genome_length=None,
                points_train=None, points_test=None, codon_consumption='eager', 
                report_items=None,
                genome_representation='list',
                stats=None, halloffame=None, 
                verbose=__debug__, run_id=None):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_, with some adaptations to run GE
    on GRAPE.
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param elite_size: The number of best individuals to be copied to the 
                    next generation.
    :params bnf_grammar, codon_size, max_tree_depth: Parameters 
                    used to mapper the individuals after crossover and
                    mutation in order to check if they are valid.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution
    """
    
    logbook = tools.Logbook()
    
    if halloffame is None:
        if elite_size != 0:
            raise ValueError("You should add a hof object to use elitism.") 
        else:
            warnings.warn('You will not register results of the best individual while not using a hof object.', hofWarning)
            logbook.header = ['gen', 'invalid'] + (stats.fields if stats else []) + ['avg_length', 'avg_nodes', 'avg_depth', 'avg_used_codons', 'invalid_count_min', 'invalid_count_avg', 'invalid_count_max', 'invalid_count_std', 'nodes_length_min', 'nodes_length_avg', 'nodes_length_max', 'nodes_length_std', 'behavioural_diversity', 'structural_diversity', 'fitness_diversity', 'selection_time', 'generation_time']
    else:
        if halloffame.maxsize < 1:
            raise ValueError("HALLOFFAME_SIZE should be greater or equal to 1")
        if elite_size > halloffame.maxsize:
            raise ValueError("HALLOFFAME_SIZE should be greater or equal to ELITE_SIZE")         
        if points_test:
            logbook.header = ['gen', 'invalid'] + (stats.fields if stats else []) + ['fitness_test', 'best_ind_length', 'avg_length', 'best_ind_nodes', 'avg_nodes', 'best_ind_depth', 'avg_depth', 'avg_used_codons', 'best_ind_used_codons', 'invalid_count_min', 'invalid_count_avg', 'invalid_count_max', 'invalid_count_std', 'nodes_length_min', 'nodes_length_avg', 'nodes_length_max', 'nodes_length_std', 'behavioural_diversity', 'structural_diversity', 'fitness_diversity', 'selection_time', 'generation_time']
        else:
            logbook.header = ['gen', 'invalid'] + (stats.fields if stats else []) + ['best_ind_length', 'avg_length', 'best_ind_nodes', 'avg_nodes', 'best_ind_depth', 'avg_depth', 'avg_used_codons', 'best_ind_used_codons', 'invalid_count_min', 'invalid_count_avg', 'invalid_count_max', 'invalid_count_std', 'nodes_length_min', 'nodes_length_avg', 'nodes_length_max', 'nodes_length_std', 'behavioural_diversity', 'structural_diversity', 'fitness_diversity', 'selection_time', 'generation_time']

    start_gen = time.time()        
    # Evaluate the individuals with an invalid fitness
    for ind in population:
        if not ind.fitness.valid:
            ind.fitness.values = toolbox.evaluate(ind, points_train)
        
    valid0 = [ind for ind in population if not ind.invalid]
    valid = [ind for ind in valid0 if not math.isnan(ind.fitness.values[0])]
    if len(valid0) != len(valid):
        warnings.warn("Warning: There are valid individuals with fitness = NaN in the population. We will avoid them.")
    invalid = len(population) - len(valid0) #We use the original number of invalids in this case, because we just want to count the completely mapped individuals
    
    # ===== MODIFICATION BY AGHIL FOR UGE INVALID INDIVIDUALS TRACKING =====
    # Added invalid individuals statistics calculation for UGE integration
    # For initial generation, we only have one invalid count, so min=max=avg=invalid
    invalid_count_min = invalid
    invalid_count_max = invalid
    invalid_count_avg = float(invalid)
    invalid_count_std = 0.0  # Standard deviation is 0 when all values are the same
    # ===== END MODIFICATION BY AGHIL =====
    
    list_structures = []
    if 'fitness_diversity' in report_items:
        list_fitnesses = []
    if 'behavioural_diversity' in report_items:
        behaviours = np.zeros([len(valid), len(valid[0].fitness_each_sample)], dtype=float)
    
    #for ind in offspring:
    for idx, ind in enumerate(valid):
        list_structures.append(str(ind.structure))
        if 'fitness_diversity' in report_items:
            list_fitnesses.append(str(ind.fitness.values[0]))
        if 'behavioural_diversity' in report_items:
            behaviours[idx, :] = ind.fitness_each_sample
            
    unique_structures = np.unique(list_structures, return_counts=False)  
    if 'fitness_diversity' in report_items:
        unique_fitnesses = np.unique(list_fitnesses, return_counts=False) 
    if 'behavioural_diversity' in report_items:
        unique_behaviours = np.unique(behaviours, axis=0)
    
    structural_diversity = len(unique_structures)/len(population)
    fitness_diversity = len(unique_fitnesses)/(len(points_train[1])+1) if 'fitness_diversity' in report_items else 0 #TODO generalise for other problems, because it only works if the fitness is proportional to the number of testcases correctly predicted
    behavioural_diversity = len(unique_behaviours)/len(population) if 'behavioural_diversity' in report_items else 0

    # Update the hall of fame with the generated individuals
    if halloffame is not None:
        halloffame.update(valid)
        best_ind_length = len(halloffame.items[0].genome) 
        best_ind_nodes = halloffame.items[0].nodes
        best_ind_depth = halloffame.items[0].depth
        best_ind_used_codons = halloffame.items[0].used_codons
        if not verbose:
            print("gen =", 0, ", Best fitness =", halloffame.items[0].fitness.values)
    
    length = [len(ind.genome) for ind in valid]
    avg_length = sum(length)/len(length)
    
    nodes = [ind.nodes for ind in valid]
    avg_nodes = sum(nodes)/len(nodes)
    
    # ===== MODIFICATION BY AGHIL FOR UGE NODES LENGTH TRACKING =====
    # Added nodes length statistics calculation for UGE integration
    nodes_length_min = min(nodes) if nodes else 0
    nodes_length_max = max(nodes) if nodes else 0
    nodes_length_avg = avg_nodes
    # Calculate standard deviation for nodes length
    if len(nodes) > 1:
        nodes_length_std = math.sqrt(sum((x - avg_nodes) ** 2 for x in nodes) / (len(nodes) - 1))
    else:
        nodes_length_std = 0.0
    # ===== END MODIFICATION BY AGHIL =====
    
    depth = [ind.depth for ind in valid]
    avg_depth = sum(depth)/len(depth)
    
    used_codons = [ind.used_codons for ind in valid]
    avg_used_codons = sum(used_codons)/len(used_codons)
    
    end_gen = time.time()
    generation_time = end_gen-start_gen
        
    selection_time = 0
    
    # Update cache context for generation 0
    if run_id:
        set_cache_context(run_id, 0)
    
    if points_test:
        fitness_test = np.nan
    
    record = stats.compile(population) if stats else {}
    if points_test: 
        logbook.record(gen=0, invalid=invalid, **record, 
                       fitness_test=fitness_test,
                       best_ind_length=best_ind_length, avg_length=avg_length, 
                       best_ind_nodes=best_ind_nodes,
                       avg_nodes=avg_nodes,
                       best_ind_depth=best_ind_depth,
                       avg_depth=avg_depth,
                       avg_used_codons=avg_used_codons,
                       best_ind_used_codons=best_ind_used_codons,
                       # ===== MODIFICATION BY AGHIL FOR UGE INVALID INDIVIDUALS TRACKING =====
                       invalid_count_min=invalid_count_min,
                       invalid_count_avg=invalid_count_avg,
                       invalid_count_max=invalid_count_max,
                       invalid_count_std=invalid_count_std,
                       # ===== END MODIFICATION BY AGHIL =====
                       # ===== MODIFICATION BY AGHIL FOR UGE NODES LENGTH TRACKING =====
                       nodes_length_min=nodes_length_min,
                       nodes_length_avg=nodes_length_avg,
                       nodes_length_max=nodes_length_max,
                       nodes_length_std=nodes_length_std,
                       # ===== END MODIFICATION BY AGHIL =====
                       behavioural_diversity=behavioural_diversity,
                       structural_diversity=structural_diversity,
                       fitness_diversity=fitness_diversity,
                       selection_time=selection_time, 
                       generation_time=generation_time)
    else:
        logbook.record(gen=0, invalid=invalid, **record, 
                       best_ind_length=best_ind_length, avg_length=avg_length, 
                       best_ind_nodes=best_ind_nodes,
                       avg_nodes=avg_nodes,
                       best_ind_depth=best_ind_depth,
                       avg_depth=avg_depth,
                       avg_used_codons=avg_used_codons,
                       best_ind_used_codons=best_ind_used_codons,
                       # ===== MODIFICATION BY AGHIL FOR UGE INVALID INDIVIDUALS TRACKING =====
                       invalid_count_min=invalid_count_min,
                       invalid_count_avg=invalid_count_avg,
                       invalid_count_max=invalid_count_max,
                       invalid_count_std=invalid_count_std,
                       # ===== END MODIFICATION BY AGHIL =====
                       # ===== MODIFICATION BY AGHIL FOR UGE NODES LENGTH TRACKING =====
                       nodes_length_min=nodes_length_min,
                       nodes_length_avg=nodes_length_avg,
                       nodes_length_max=nodes_length_max,
                       nodes_length_std=nodes_length_std,
                       # ===== END MODIFICATION BY AGHIL =====
                       behavioural_diversity=behavioural_diversity,
                       structural_diversity=structural_diversity,
                       fitness_diversity=fitness_diversity,
                       selection_time=selection_time, 
                       generation_time=generation_time)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(logbook.select("gen")[-1]+1, ngen + 1):
        start_gen = time.time()
        
        # Update cache context for FEC tracking
        if run_id:
            set_cache_context(run_id, gen)
    
        # Select the next generation individuals
        start = time.time()    
        offspring = toolbox.select(valid, len(population)-elite_size)
        end = time.time()
        selection_time = end-start
        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb,
                           bnf_grammar, codon_size, max_tree_depth, 
                           codon_consumption, genome_representation,
                           max_genome_length)

        # Evaluate the individuals with an invalid fitness
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind, points_train)
                
        #Update population for next generation
        population[:] = offspring
        #Include in the population the elitist individuals
        for i in range(elite_size):
            population.append(halloffame.items[i])
            
        valid0 = [ind for ind in population if not ind.invalid]
        valid = [ind for ind in valid0 if not math.isnan(ind.fitness.values[0])]
        if len(valid0) != len(valid):
            warnings.warn("Warning: There are valid individuals with fitness = NaN in the population. We will avoid in the statistics.")
        invalid = len(population) - len(valid0) #We use the original number of invalids in this case, because we just want to count the completely mapped individuals
        
        # ===== MODIFICATION BY AGHIL FOR UGE INVALID INDIVIDUALS TRACKING =====
        # Added invalid individuals statistics calculation for UGE integration
        # For each generation, we only have one invalid count, so min=max=avg=invalid
        invalid_count_min = invalid
        invalid_count_max = invalid
        invalid_count_avg = float(invalid)
        invalid_count_std = 0.0  # Standard deviation is 0 when all values are the same
        # ===== END MODIFICATION BY AGHIL =====
        
        list_structures = []
        if 'fitness_diversity' in report_items:
            list_fitnesses = []
        if 'behavioural_diversity' in report_items:
            behaviours = np.zeros([len(valid), len(valid[0].fitness_each_sample)], dtype=float)
        
        for idx, ind in enumerate(valid):
            list_structures.append(str(ind.structure))
            if 'fitness_diversity' in report_items:
                list_fitnesses.append(str(ind.fitness.values[0]))
            if 'behavioural_diversity' in report_items:
                behaviours[idx, :] = ind.fitness_each_sample
                
        unique_structures = np.unique(list_structures, return_counts=False)  
        if 'fitness_diversity' in report_items:
            unique_fitnesses = np.unique(list_fitnesses, return_counts=False) 
        if 'behavioural_diversity' in report_items:
            unique_behaviours = np.unique(behaviours, axis=0)
        
        structural_diversity = len(unique_structures)/len(population)
        fitness_diversity = len(unique_fitnesses)/(len(points_train[1])+1) if 'fitness_diversity' in report_items else 0 #TODO generalise for other problems, because it only works if the fitness is proportional to the number of testcases correctly predicted
        behavioural_diversity = len(unique_behaviours)/len(population) if 'behavioural_diversity' in report_items else 0
        
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(valid)
            best_ind_length = len(halloffame.items[0].genome)
            best_ind_nodes = halloffame.items[0].nodes
            best_ind_depth = halloffame.items[0].depth
            best_ind_used_codons = halloffame.items[0].used_codons
            if not verbose:
                print("gen =", gen, ", Best fitness =", halloffame.items[0].fitness.values, ", Number of invalids =", invalid)
            if points_test:
                # Calculate test fitness for the best individual in each generation
                fitness_test = toolbox.evaluate(halloffame.items[0], points_test)[0]
            
        length = [len(ind.genome) for ind in valid]
        avg_length = sum(length)/len(length)
        
        nodes = [ind.nodes for ind in valid]
        avg_nodes = sum(nodes)/len(nodes)
        
        # ===== MODIFICATION BY AGHIL FOR UGE NODES LENGTH TRACKING =====
        # Added nodes length statistics calculation for UGE integration
        nodes_length_min = min(nodes) if nodes else 0
        nodes_length_max = max(nodes) if nodes else 0
        nodes_length_avg = avg_nodes
        # Calculate standard deviation for nodes length
        if len(nodes) > 1:
            nodes_length_std = math.sqrt(sum((x - avg_nodes) ** 2 for x in nodes) / (len(nodes) - 1))
        else:
            nodes_length_std = 0.0
        # ===== END MODIFICATION BY AGHIL =====
        
        depth = [ind.depth for ind in valid]
        avg_depth = sum(depth)/len(depth)
        
        used_codons = [ind.used_codons for ind in valid]
        avg_used_codons = sum(used_codons)/len(used_codons)
        
        end_gen = time.time()
        generation_time = end_gen-start_gen
        
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        if points_test: 
            logbook.record(gen=gen, invalid=invalid, **record, 
                       fitness_test=fitness_test,
                       best_ind_length=best_ind_length, avg_length=avg_length, 
                       best_ind_nodes=best_ind_nodes,
                       avg_nodes=avg_nodes,
                       best_ind_depth=best_ind_depth,
                       avg_depth=avg_depth,
                       avg_used_codons=avg_used_codons,
                       best_ind_used_codons=best_ind_used_codons,
                       # ===== MODIFICATION BY AGHIL FOR UGE INVALID INDIVIDUALS TRACKING =====
                       invalid_count_min=invalid_count_min,
                       invalid_count_avg=invalid_count_avg,
                       invalid_count_max=invalid_count_max,
                       invalid_count_std=invalid_count_std,
                       # ===== END MODIFICATION BY AGHIL =====
                       # ===== MODIFICATION BY AGHIL FOR UGE NODES LENGTH TRACKING =====
                       nodes_length_min=nodes_length_min,
                       nodes_length_avg=nodes_length_avg,
                       nodes_length_max=nodes_length_max,
                       nodes_length_std=nodes_length_std,
                       # ===== END MODIFICATION BY AGHIL =====
                       behavioural_diversity=behavioural_diversity,
                       structural_diversity=structural_diversity,
                       fitness_diversity=fitness_diversity,
                       selection_time=selection_time, 
                       generation_time=generation_time)
        else:
            logbook.record(gen=gen, invalid=invalid, **record, 
                       best_ind_length=best_ind_length, avg_length=avg_length, 
                       best_ind_nodes=best_ind_nodes,
                       avg_nodes=avg_nodes,
                       best_ind_depth=best_ind_depth,
                       avg_depth=avg_depth,
                       avg_used_codons=avg_used_codons,
                       best_ind_used_codons=best_ind_used_codons,
                       # ===== MODIFICATION BY AGHIL FOR UGE INVALID INDIVIDUALS TRACKING =====
                       invalid_count_min=invalid_count_min,
                       invalid_count_avg=invalid_count_avg,
                       invalid_count_max=invalid_count_max,
                       invalid_count_std=invalid_count_std,
                       # ===== END MODIFICATION BY AGHIL =====
                       # ===== MODIFICATION BY AGHIL FOR UGE NODES LENGTH TRACKING =====
                       nodes_length_min=nodes_length_min,
                       nodes_length_avg=nodes_length_avg,
                       nodes_length_max=nodes_length_max,
                       nodes_length_std=nodes_length_std,
                       # ===== END MODIFICATION BY AGHIL =====
                       behavioural_diversity=behavioural_diversity,
                       structural_diversity=structural_diversity,
                       fitness_diversity=fitness_diversity,
                       selection_time=selection_time, 
                       generation_time=generation_time)
                
        if verbose:
            print(logbook.stream)

    return population, logbook


def ge_eaSimpleWithElitism_fec(population, toolbox, cxpb, mutpb, ngen, elite_size, 
                bnf_grammar, codon_size, max_tree_depth, 
                max_genome_length=None,
                points_train=None, points_test=None, codon_consumption='eager', 
                report_items=None,
                genome_representation='list',
                stats=None, halloffame=None, 
                verbose=__debug__, run_id=None, fec_cache=None, phenotype_tracker=None):
    """Same as ge_eaSimpleWithElitism but with FEC (Fitness Evaluation Cache) tracking.
    
    This function is identical to ge_eaSimpleWithElitism but adds per-generation tracking
    for cache hits and fake hits by calling fec_cache.start_generation() and 
    fec_cache.end_generation() at appropriate points.
    
    All parameters are the same as ge_eaSimpleWithElitism, with the addition of:
    :param fec_cache: FECCache instance for tracking cache statistics per generation
    :param phenotype_tracker: PhenotypeTracker instance for tracking phenotypes
    """
    # Copy of ge_eaSimpleWithElitism with FEC tracking added
    logbook = tools.Logbook()
    
    if halloffame is None:
        if elite_size != 0:
            raise ValueError("You should add a hof object to use elitism.") 
        else:
            warnings.warn('You will not register results of the best individual while not using a hof object.', hofWarning)
            logbook.header = ['gen', 'invalid'] + (stats.fields if stats else []) + ['avg_length', 'avg_nodes', 'avg_depth', 'avg_used_codons', 'invalid_count_min', 'invalid_count_avg', 'invalid_count_max', 'invalid_count_std', 'nodes_length_min', 'nodes_length_avg', 'nodes_length_max', 'nodes_length_std', 'selection_time', 'generation_time']
    else:
        if halloffame.maxsize < 1:
            raise ValueError("HALLOFFAME_SIZE should be greater or equal to 1")
        if elite_size > halloffame.maxsize:
            raise ValueError("HALLOFFAME_SIZE should be greater or equal to ELITE_SIZE")         
        if points_test:
            logbook.header = ['gen', 'invalid'] + (stats.fields if stats else []) + ['fitness_test', 'best_ind_length', 'avg_length', 'best_ind_nodes', 'avg_nodes', 'best_ind_depth', 'avg_depth', 'avg_used_codons', 'best_ind_used_codons', 'invalid_count_min', 'invalid_count_avg', 'invalid_count_max', 'invalid_count_std', 'nodes_length_min', 'nodes_length_avg', 'nodes_length_max', 'nodes_length_std', 'selection_time', 'generation_time']
        else:
            logbook.header = ['gen', 'invalid'] + (stats.fields if stats else []) + ['best_ind_length', 'avg_length', 'best_ind_nodes', 'avg_nodes', 'best_ind_depth', 'avg_depth', 'avg_used_codons', 'best_ind_used_codons', 'invalid_count_min', 'invalid_count_avg', 'invalid_count_max', 'invalid_count_std', 'nodes_length_min', 'nodes_length_avg', 'nodes_length_max', 'nodes_length_std', 'selection_time', 'generation_time']

    # Start generation 0 tracking for FEC
    if fec_cache is not None:
        fec_cache.start_generation()
    
    start_gen = time.time()        
    # Evaluate the individuals with an invalid fitness
    for ind in population:
        if not ind.fitness.valid:
            ind.fitness.values = toolbox.evaluate(ind, points_train)
        
    valid0 = [ind for ind in population if not ind.invalid]
    valid = [ind for ind in valid0 if not math.isnan(ind.fitness.values[0])]
    if len(valid0) != len(valid):
        warnings.warn("Warning: There are valid individuals with fitness = NaN in the population. We will avoid them.")
    invalid = len(population) - len(valid0)
    
    invalid_count_min = invalid
    invalid_count_max = invalid
    invalid_count_avg = float(invalid)
    invalid_count_std = 0.0
    
    # Track phenotypes for generation 0
    if phenotype_tracker is not None:
        phenotype_tracker.record_generation(population, halloffame, elite_size)
    
    if halloffame is not None:
        halloffame.update(valid)
        if len(halloffame) > 0:
            best_ind_length = len(halloffame.items[0].genome) 
            best_ind_nodes = halloffame.items[0].nodes
            best_ind_depth = halloffame.items[0].depth
            best_ind_used_codons = halloffame.items[0].used_codons
            if not verbose:
                print("gen =", 0, ", Best fitness =", halloffame.items[0].fitness.values)
        else:
            best_ind_length = 0
            best_ind_nodes = 0
            best_ind_depth = 0
            best_ind_used_codons = 0
    else:
        best_ind_length = 0
        best_ind_nodes = 0
        best_ind_depth = 0
        best_ind_used_codons = 0
    
    length = [len(ind.genome) for ind in valid]
    avg_length = sum(length)/len(length) if len(length) > 0 else 0.0
    
    nodes = [ind.nodes for ind in valid]
    avg_nodes = sum(nodes)/len(nodes) if len(nodes) > 0 else 0.0
    
    nodes_length_min = min(nodes) if nodes else 0
    nodes_length_max = max(nodes) if nodes else 0
    nodes_length_avg = avg_nodes
    if len(nodes) > 1:
        nodes_length_std = math.sqrt(sum((x - avg_nodes) ** 2 for x in nodes) / (len(nodes) - 1))
    else:
        nodes_length_std = 0.0
    
    depth = [ind.depth for ind in valid]
    avg_depth = sum(depth)/len(depth) if len(depth) > 0 else 0.0
    
    used_codons = [ind.used_codons for ind in valid]
    avg_used_codons = sum(used_codons)/len(used_codons) if len(used_codons) > 0 else 0.0
    
    end_gen = time.time()
    generation_time = end_gen-start_gen
        
    selection_time = 0
    
    if run_id:
        set_cache_context(run_id, 0)
    
    if points_test:
        fitness_test = np.nan
    
    record = stats.compile(population) if stats else {}
    if points_test: 
        logbook.record(gen=0, invalid=invalid, **record, 
                       fitness_test=fitness_test,
                       best_ind_length=best_ind_length, avg_length=avg_length, 
                       best_ind_nodes=best_ind_nodes,
                       avg_nodes=avg_nodes,
                       best_ind_depth=best_ind_depth,
                       avg_depth=avg_depth,
                       avg_used_codons=avg_used_codons,
                       best_ind_used_codons=best_ind_used_codons,
                       invalid_count_min=invalid_count_min,
                       invalid_count_avg=invalid_count_avg,
                       invalid_count_max=invalid_count_max,
                       invalid_count_std=invalid_count_std,
                       nodes_length_min=nodes_length_min,
                       nodes_length_avg=nodes_length_avg,
                       nodes_length_max=nodes_length_max,
                       nodes_length_std=nodes_length_std,
                       selection_time=selection_time, 
                       generation_time=generation_time)
    else:
        logbook.record(gen=0, invalid=invalid, **record, 
                       best_ind_length=best_ind_length, avg_length=avg_length, 
                       best_ind_nodes=best_ind_nodes,
                       avg_nodes=avg_nodes,
                       best_ind_depth=best_ind_depth,
                       avg_depth=avg_depth,
                       avg_used_codons=avg_used_codons,
                       best_ind_used_codons=best_ind_used_codons,
                       invalid_count_min=invalid_count_min,
                       invalid_count_avg=invalid_count_avg,
                       invalid_count_max=invalid_count_max,
                       invalid_count_std=invalid_count_std,
                       nodes_length_min=nodes_length_min,
                       nodes_length_avg=nodes_length_avg,
                       nodes_length_max=nodes_length_max,
                       nodes_length_std=nodes_length_std,
                       selection_time=selection_time, 
                       generation_time=generation_time)
    
    # End generation 0 tracking for FEC
    if fec_cache is not None:
        fec_cache.end_generation()
    
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(logbook.select("gen")[-1]+1, ngen + 1):
        # Start generation tracking for FEC
        if fec_cache is not None:
            fec_cache.start_generation()
        
        start_gen = time.time()
        
        if run_id:
            set_cache_context(run_id, gen)
    
        start = time.time()    
        offspring = toolbox.select(valid, len(population)-elite_size)
        end = time.time()
        selection_time = end-start
        offspring = varAnd(offspring, toolbox, cxpb, mutpb,
                           bnf_grammar, codon_size, max_tree_depth, 
                           codon_consumption, genome_representation,
                           max_genome_length)

        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind, points_train)
                
        population[:] = offspring
        for i in range(elite_size):
            population.append(halloffame.items[i])
            
        valid0 = [ind for ind in population if not ind.invalid]
        valid = [ind for ind in valid0 if not math.isnan(ind.fitness.values[0])]
        if len(valid0) != len(valid):
            warnings.warn("Warning: There are valid individuals with fitness = NaN in the population. We will avoid in the statistics.")
        invalid = len(population) - len(valid0)
        
        invalid_count_min = invalid
        invalid_count_max = invalid
        invalid_count_avg = float(invalid)
        invalid_count_std = 0.0
        
        if halloffame is not None:
            halloffame.update(valid)
            best_ind_length = len(halloffame.items[0].genome)
            best_ind_nodes = halloffame.items[0].nodes
            best_ind_depth = halloffame.items[0].depth
            best_ind_used_codons = halloffame.items[0].used_codons
            if not verbose:
                print("gen =", gen, ", Best fitness =", halloffame.items[0].fitness.values, ", Number of invalids =", invalid)
            if points_test:
                fitness_test = toolbox.evaluate(halloffame.items[0], points_test)[0]
        
        # Track phenotypes for this generation
        if phenotype_tracker is not None:
            phenotype_tracker.record_generation(population, halloffame, elite_size)
            
        length = [len(ind.genome) for ind in valid]
        avg_length = sum(length)/len(length) if len(length) > 0 else 0.0
        
        nodes = [ind.nodes for ind in valid]
        avg_nodes = sum(nodes)/len(nodes) if len(nodes) > 0 else 0.0
        
        nodes_length_min = min(nodes) if nodes else 0
        nodes_length_max = max(nodes) if nodes else 0
        nodes_length_avg = avg_nodes
        if len(nodes) > 1:
            nodes_length_std = math.sqrt(sum((x - avg_nodes) ** 2 for x in nodes) / (len(nodes) - 1))
        else:
            nodes_length_std = 0.0
        
        depth = [ind.depth for ind in valid]
        avg_depth = sum(depth)/len(depth) if len(depth) > 0 else 0.0
        
        used_codons = [ind.used_codons for ind in valid]
        avg_used_codons = sum(used_codons)/len(used_codons) if len(used_codons) > 0 else 0.0
        
        end_gen = time.time()
        generation_time = end_gen-start_gen
        
        # End generation tracking for FEC
        if fec_cache is not None:
            fec_cache.end_generation()
        
        record = stats.compile(population) if stats else {}
        if points_test: 
            logbook.record(gen=gen, invalid=invalid, **record, 
                       fitness_test=fitness_test,
                       best_ind_length=best_ind_length, avg_length=avg_length, 
                       best_ind_nodes=best_ind_nodes,
                       avg_nodes=avg_nodes,
                       best_ind_depth=best_ind_depth,
                       avg_depth=avg_depth,
                       avg_used_codons=avg_used_codons,
                       best_ind_used_codons=best_ind_used_codons,
                       invalid_count_min=invalid_count_min,
                       invalid_count_avg=invalid_count_avg,
                       invalid_count_max=invalid_count_max,
                       invalid_count_std=invalid_count_std,
                       nodes_length_min=nodes_length_min,
                       nodes_length_avg=nodes_length_avg,
                       nodes_length_max=nodes_length_max,
                       nodes_length_std=nodes_length_std,
                       selection_time=selection_time, 
                       generation_time=generation_time)
        else:
            logbook.record(gen=gen, invalid=invalid, **record, 
                       best_ind_length=best_ind_length, avg_length=avg_length, 
                       best_ind_nodes=best_ind_nodes,
                       avg_nodes=avg_nodes,
                       best_ind_depth=best_ind_depth,
                       avg_depth=avg_depth,
                       avg_used_codons=avg_used_codons,
                       best_ind_used_codons=best_ind_used_codons,
                       invalid_count_min=invalid_count_min,
                       invalid_count_avg=invalid_count_avg,
                       invalid_count_max=invalid_count_max,
                       invalid_count_std=invalid_count_std,
                       nodes_length_min=nodes_length_min,
                       nodes_length_avg=nodes_length_avg,
                       nodes_length_max=nodes_length_max,
                       nodes_length_std=nodes_length_std,
                       selection_time=selection_time, 
                       generation_time=generation_time)
                
        if verbose:
            print(logbook.stream)

    return population, logbook


def ge_eaSimpleWithElitism_dynamic(population, toolbox, ngen,
                bnf_grammar,
                points_train=None, points_test=None,
                report_items=None,
                stats=None, halloffame=None,
                verbose=__debug__,
                parameter_configs=None,
                random_seed=42,
                run_id=None):
    """Same as ge_eaSimpleWithElitism, but updates configuration each generation.

    Parameter values are calculated per generation using parameter_configs. Only
    'fixed' and 'custom' modes are supported. Custom mode uses:
      - change_every, change_amount, change_operation ('add'|'subtract')
      - min_value, max_value for clamping
    """

    def _gen_value(key, default_val, generation):
        cfg = (parameter_configs or {}).get(key, {'mode': 'fixed', 'value': default_val})
        mode = cfg.get('mode', 'fixed')
        base_val = cfg.get('value', default_val)
        if mode == 'custom':
            change_every = cfg.get('change_every', 5)
            change_amount = cfg.get('change_amount', 1)
            op = cfg.get('change_operation', 'add')
            min_v = cfg.get('min_value', default_val)
            max_v = cfg.get('max_value', default_val)
            cycles = generation // max(1, int(change_every))
            try:
                new_val = base_val + (change_amount * cycles if op == 'add' else -change_amount * cycles)
            except TypeError:
                # Non-numeric (categorical) -> treat as fixed
                return base_val
            # Clamp and preserve type for ints
            try:
                new_val = max(min_v, min(max_v, new_val))
            except TypeError:
                return base_val
            if isinstance(default_val, int):
                return int(new_val)
            return float(new_val) if isinstance(default_val, float) else new_val
        # fixed or unsupported -> fixed
        return base_val

    # Get default values from parameter_configs or use reasonable defaults
    default_cxpb = _gen_value('p_crossover', 0.8, 0)
    default_mutpb = _gen_value('p_mutation', 0.01, 0)
    default_elite_size = int(_gen_value('elite_size', 1, 0))
    default_tournsize = int(_gen_value('tournsize', 7, 0))
    default_codon_size = int(_gen_value('codon_size', 255, 0))
    default_max_tree_depth = int(_gen_value('max_tree_depth', 35, 0))
    default_codon_consumption = _gen_value('codon_consumption', 'eager', 0)
    default_genome_representation = _gen_value('genome_representation', 'list', 0)
    default_max_genome_length = _gen_value('max_init_genome_length', None, 0)

    random.seed(random_seed)
    np.random.seed(random_seed)

    logbook = tools.Logbook()

    if halloffame is None:
        if default_elite_size != 0:
            raise ValueError("You should add a hof object to use elitism.")
        else:
            warnings.warn('You will not register results of the best individual while not using a hof object.', hofWarning)
            logbook.header = ['gen', 'invalid'] + (stats.fields if stats else []) + ['avg_length', 'avg_nodes', 'avg_depth', 'avg_used_codons', 'invalid_count_min', 'invalid_count_avg', 'invalid_count_max', 'invalid_count_std', 'nodes_length_min', 'nodes_length_avg', 'nodes_length_max', 'nodes_length_std', 'behavioural_diversity', 'structural_diversity', 'fitness_diversity', 'selection_time', 'generation_time']
    else:
        if halloffame.maxsize < 1:
            raise ValueError("HALLOFFAME_SIZE should be greater or equal to 1")
        if default_elite_size > halloffame.maxsize:
            raise ValueError("HALLOFFAME_SIZE should be greater or equal to ELITE_SIZE")
        if points_test:
            logbook.header = ['gen', 'invalid'] + (stats.fields if stats else []) + ['fitness_test', 'best_ind_length', 'avg_length', 'best_ind_nodes', 'avg_nodes', 'best_ind_depth', 'avg_depth', 'avg_used_codons', 'best_ind_used_codons', 'invalid_count_min', 'invalid_count_avg', 'invalid_count_max', 'invalid_count_std', 'nodes_length_min', 'nodes_length_avg', 'nodes_length_max', 'nodes_length_std', 'behavioural_diversity', 'structural_diversity', 'fitness_diversity', 'selection_time', 'generation_time']
        else:
            logbook.header = ['gen', 'invalid'] + (stats.fields if stats else []) + ['best_ind_length', 'avg_length', 'best_ind_nodes', 'avg_nodes', 'best_ind_depth', 'avg_depth', 'avg_used_codons', 'best_ind_used_codons', 'invalid_count_min', 'invalid_count_avg', 'invalid_count_max', 'invalid_count_std', 'nodes_length_min', 'nodes_length_avg', 'nodes_length_max', 'nodes_length_std', 'behavioural_diversity', 'structural_diversity', 'fitness_diversity', 'selection_time', 'generation_time']

    start_gen = time.time()
    # Evaluate invalid fitness
    for ind in population:
        if not ind.fitness.valid:
            ind.fitness.values = toolbox.evaluate(ind, points_train)

    valid0 = [ind for ind in population if not ind.invalid]
    valid = [ind for ind in valid0 if not math.isnan(ind.fitness.values[0])]
    if len(valid0) != len(valid):
        warnings.warn("Warning: There are valid individuals with fitness = NaN in the population. We will avoid them.")
    invalid = len(population) - len(valid0)

    # Initial stats
    invalid_count_min = invalid
    invalid_count_max = invalid
    invalid_count_avg = float(invalid)
    invalid_count_std = 0.0

    list_structures = []
    if 'fitness_diversity' in report_items:
        list_fitnesses = []
    if 'behavioural_diversity' in report_items:
        behaviours = np.zeros([len(valid), len(valid[0].fitness_each_sample)], dtype=float)

    for idx, ind in enumerate(valid):
        list_structures.append(str(ind.structure))
        if 'fitness_diversity' in report_items:
            list_fitnesses.append(str(ind.fitness.values[0]))
        if 'behavioural_diversity' in report_items:
            behaviours[idx, :] = ind.fitness_each_sample

    unique_structures = np.unique(list_structures, return_counts=False)
    if 'fitness_diversity' in report_items:
        unique_fitnesses = np.unique(list_fitnesses, return_counts=False)
    if 'behavioural_diversity' in report_items:
        unique_behaviours = np.unique(behaviours, axis=0)

    structural_diversity = len(unique_structures)/len(population)
    fitness_diversity = len(unique_fitnesses)/(len(points_train[1])+1) if 'fitness_diversity' in report_items else 0
    behavioural_diversity = len(unique_behaviours)/len(population) if 'behavioural_diversity' in report_items else 0

    if halloffame is not None:
        halloffame.update(valid)
        best_ind_length = len(halloffame.items[0].genome)
        best_ind_nodes = halloffame.items[0].nodes
        best_ind_depth = halloffame.items[0].depth
        best_ind_used_codons = halloffame.items[0].used_codons
        if not verbose:
            print("gen =", 0, ", Best fitness =", halloffame.items[0].fitness.values)

    length = [len(ind.genome) for ind in valid]
    avg_length = sum(length)/len(length)
    nodes = [ind.nodes for ind in valid]
    avg_nodes = sum(nodes)/len(nodes)
    nodes_length_min = min(nodes) if nodes else 0
    nodes_length_max = max(nodes) if nodes else 0
    nodes_length_avg = avg_nodes
    if len(nodes) > 1:
        nodes_length_std = math.sqrt(sum((x - avg_nodes) ** 2 for x in nodes) / (len(nodes) - 1))
    else:
        nodes_length_std = 0.0

    depth = [ind.depth for ind in valid]
    avg_depth = sum(depth)/len(depth)
    used_codons = [ind.used_codons for ind in valid]
    avg_used_codons = sum(used_codons)/len(used_codons)

    end_gen = time.time()
    generation_time = end_gen-start_gen
    selection_time = 0
    
    # Update cache context for generation 0
    if run_id:
        set_cache_context(run_id, 0)
    
    if points_test:
        fitness_test = np.nan

    record = stats.compile(population) if stats else {}
    if points_test:
        logbook.record(gen=0, invalid=invalid, **record,
                       fitness_test=fitness_test,
                       best_ind_length=best_ind_length, avg_length=avg_length,
                       best_ind_nodes=best_ind_nodes,
                       avg_nodes=avg_nodes,
                       best_ind_depth=best_ind_depth,
                       avg_depth=avg_depth,
                       avg_used_codons=avg_used_codons,
                       best_ind_used_codons=best_ind_used_codons,
                       invalid_count_min=invalid_count_min,
                       invalid_count_avg=invalid_count_avg,
                       invalid_count_max=invalid_count_max,
                       invalid_count_std=invalid_count_std,
                       nodes_length_min=nodes_length_min,
                       nodes_length_avg=nodes_length_avg,
                       nodes_length_max=nodes_length_max,
                       nodes_length_std=nodes_length_std,
                       behavioural_diversity=behavioural_diversity,
                       structural_diversity=structural_diversity,
                       fitness_diversity=fitness_diversity,
                       selection_time=selection_time,
                       generation_time=generation_time)
    else:
        logbook.record(gen=0, invalid=invalid, **record,
                       best_ind_length=best_ind_length, avg_length=avg_length,
                       best_ind_nodes=best_ind_nodes,
                       avg_nodes=avg_nodes,
                       best_ind_depth=best_ind_depth,
                       avg_depth=avg_depth,
                       avg_used_codons=avg_used_codons,
                       best_ind_used_codons=best_ind_used_codons,
                       invalid_count_min=invalid_count_min,
                       invalid_count_avg=invalid_count_avg,
                       invalid_count_max=invalid_count_max,
                       invalid_count_std=invalid_count_std,
                       nodes_length_min=nodes_length_min,
                       nodes_length_avg=nodes_length_avg,
                       nodes_length_max=nodes_length_max,
                       nodes_length_std=nodes_length_std,
                       behavioural_diversity=behavioural_diversity,
                       structural_diversity=structural_diversity,
                       fitness_diversity=fitness_diversity,
                       selection_time=selection_time,
                       generation_time=generation_time)
    if verbose:
        print(logbook.stream)

    # Generational process with per-generation configuration
    for gen in range(logbook.select("gen")[-1]+1, ngen + 1):
        start_gen = time.time()
        
        # Update cache context for FEC tracking
        if run_id:
            set_cache_context(run_id, gen)

        # Compute per-generation parameters
        cur_elite_size = int(_gen_value('elite_size', default_elite_size, gen))
        cur_tournsize = int(_gen_value('tournsize', default_tournsize, gen))
        cur_cxpb = _gen_value('p_crossover', default_cxpb, gen)
        cur_mutpb = _gen_value('p_mutation', default_mutpb, gen)
        cur_max_tree_depth = int(_gen_value('max_tree_depth', default_max_tree_depth, gen))
        cur_codon_size = int(_gen_value('codon_size', default_codon_size, gen))
        cur_codon_consumption = _gen_value('codon_consumption', default_codon_consumption, gen)
        cur_genome_representation = _gen_value('genome_representation', default_genome_representation, gen)
        cur_max_genome_length = int(_gen_value('max_init_genome_length', default_max_genome_length if default_max_genome_length is not None else 0, gen)) if default_max_genome_length is not None else None

        # Selection with dynamic tournsize and elite size
        start = time.time()
        offspring = tools.selTournament(valid, len(population) - cur_elite_size, tournsize=cur_tournsize)
        end = time.time()
        selection_time = end - start

        # Variation with dynamic parameters
        offspring = varAnd(offspring, toolbox, cur_cxpb, cur_mutpb,
                           bnf_grammar, cur_codon_size, cur_max_tree_depth,
                           cur_codon_consumption, cur_genome_representation,
                           cur_max_genome_length)

        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind, points_train)

        # Update population
        population[:] = offspring
        # Include elitist individuals (clamp to available HOF size)
        if halloffame is not None and halloffame.items:
            for i in range(min(cur_elite_size, len(halloffame.items))):
                population.append(halloffame.items[i])

        valid0 = [ind for ind in population if not ind.invalid]
        valid = [ind for ind in valid0 if not math.isnan(ind.fitness.values[0])]
        if len(valid0) != len(valid):
            warnings.warn("Warning: There are valid individuals with fitness = NaN in the population. We will avoid in the statistics.")
        invalid = len(population) - len(valid0)

        # Invalid stats
        invalid_count_min = invalid
        invalid_count_max = invalid
        invalid_count_avg = float(invalid)
        invalid_count_std = 0.0

        list_structures = []
        if 'fitness_diversity' in report_items:
            list_fitnesses = []
        if 'behavioural_diversity' in report_items:
            behaviours = np.zeros([len(valid), len(valid[0].fitness_each_sample)], dtype=float)

        for idx, ind in enumerate(valid):
            list_structures.append(str(ind.structure))
            if 'fitness_diversity' in report_items:
                list_fitnesses.append(str(ind.fitness.values[0]))
            if 'behavioural_diversity' in report_items:
                behaviours[idx, :] = ind.fitness_each_sample

        unique_structures = np.unique(list_structures, return_counts=False)
        if 'fitness_diversity' in report_items:
            unique_fitnesses = np.unique(list_fitnesses, return_counts=False)
        if 'behavioural_diversity' in report_items:
            unique_behaviours = np.unique(behaviours, axis=0)

        structural_diversity = len(unique_structures)/len(population)
        fitness_diversity = len(unique_fitnesses)/(len(points_train[1])+1) if 'fitness_diversity' in report_items else 0
        behavioural_diversity = len(unique_behaviours)/len(population) if 'behavioural_diversity' in report_items else 0

        if halloffame is not None:
            halloffame.update(valid)
            best_ind_length = len(halloffame.items[0].genome)
            best_ind_nodes = halloffame.items[0].nodes
            best_ind_depth = halloffame.items[0].depth
            best_ind_used_codons = halloffame.items[0].used_codons
            if not verbose:
                print("gen =", gen, ", Best fitness =", halloffame.items[0].fitness.values, ", Number of invalids =", invalid)
            if points_test:
                fitness_test = toolbox.evaluate(halloffame.items[0], points_test)[0]

        length = [len(ind.genome) for ind in valid]
        avg_length = sum(length)/len(length)
        nodes = [ind.nodes for ind in valid]
        avg_nodes = sum(nodes)/len(nodes)
        nodes_length_min = min(nodes) if nodes else 0
        nodes_length_max = max(nodes) if nodes else 0
        nodes_length_avg = avg_nodes
        if len(nodes) > 1:
            nodes_length_std = math.sqrt(sum((x - avg_nodes) ** 2 for x in nodes) / (len(nodes) - 1))
        else:
            nodes_length_std = 0.0

        depth = [ind.depth for ind in valid]
        avg_depth = sum(depth)/len(depth)
        used_codons = [ind.used_codons for ind in valid]
        avg_used_codons = sum(used_codons)/len(used_codons)

        end_gen = time.time()
        generation_time = end_gen - start_gen

        record = stats.compile(population) if stats else {}
        if points_test:
            logbook.record(gen=gen, invalid=invalid, **record,
                           fitness_test=fitness_test,
                           best_ind_length=best_ind_length, avg_length=avg_length,
                           best_ind_nodes=best_ind_nodes,
                           avg_nodes=avg_nodes,
                           best_ind_depth=best_ind_depth,
                           avg_depth=avg_depth,
                           avg_used_codons=avg_used_codons,
                           best_ind_used_codons=best_ind_used_codons,
                           invalid_count_min=invalid_count_min,
                           invalid_count_avg=invalid_count_avg,
                           invalid_count_max=invalid_count_max,
                           invalid_count_std=invalid_count_std,
                           nodes_length_min=nodes_length_min,
                           nodes_length_avg=nodes_length_avg,
                           nodes_length_max=nodes_length_max,
                           nodes_length_std=nodes_length_std,
                           behavioural_diversity=behavioural_diversity,
                           structural_diversity=structural_diversity,
                           fitness_diversity=fitness_diversity,
                           selection_time=selection_time,
                           generation_time=generation_time)
        else:
            logbook.record(gen=gen, invalid=invalid, **record,
                           best_ind_length=best_ind_length, avg_length=avg_length,
                           best_ind_nodes=best_ind_nodes,
                           avg_nodes=avg_nodes,
                           best_ind_depth=best_ind_depth,
                           avg_depth=avg_depth,
                           avg_used_codons=avg_used_codons,
                           best_ind_used_codons=best_ind_used_codons,
                           invalid_count_min=invalid_count_min,
                           invalid_count_avg=invalid_count_avg,
                           invalid_count_max=invalid_count_max,
                           invalid_count_std=invalid_count_std,
                           nodes_length_min=nodes_length_min,
                           nodes_length_avg=nodes_length_avg,
                           nodes_length_max=nodes_length_max,
                           nodes_length_std=nodes_length_std,
                           behavioural_diversity=behavioural_diversity,
                           structural_diversity=structural_diversity,
                           fitness_diversity=fitness_diversity,
                           selection_time=selection_time,
                           generation_time=generation_time)

        if verbose:
            print(logbook.stream)

    return population, logbook