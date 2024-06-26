"""
Makes possible reporter classes,
which are triggered on particular events and may provide information to the user,
may do something else such as checkpointing, or may do both.
"""
from __future__ import division, print_function

import time

from neat.math_util import mean, stdev
from neat.six_util import itervalues, iterkeys

# TODO: Add a curses-based reporter.


class ReporterSet(object):
    """
    Keeps track of the set of reporters
    and gives methods to dispatch them at appropriate points.
    """
    def __init__(self):
        self.reporters = []

    def add(self, reporter):
        self.reporters.append(reporter)

    def remove(self, reporter):
        self.reporters.remove(reporter)

    def start_generation(self, gen):
        for r in self.reporters:
            r.start_generation(gen)

    def end_generation(self, config, population, species_set):
        for r in self.reporters:
            r.end_generation(config, population, species_set)

    def post_evaluate(self, config, population, species, best_genome):
        for r in self.reporters:
            r.post_evaluate(config, population, species, best_genome)

    def post_reproduction(self, config, population, species):
        for r in self.reporters:
            r.post_reproduction(config, population, species)

    def complete_extinction(self):
        for r in self.reporters:
            r.complete_extinction()

    def found_solution(self, config, generation, best):
        for r in self.reporters:
            r.found_solution(config, generation, best)

    def species_stagnant(self, sid, species):
        for r in self.reporters:
            r.species_stagnant(sid, species)

    def info(self, msg):
        for r in self.reporters:
            r.info(msg)


class BaseReporter(object):
    """Definition of the reporter interface expected by ReporterSet."""
    def start_generation(self, generation):
        pass

    def end_generation(self, config, population, species_set):
        pass

    def post_evaluate(self, config, population, species, best_genome):
        pass

    def post_reproduction(self, config, population, species):
        pass

    def complete_extinction(self):
        pass

    def found_solution(self, config, generation, best):
        pass

    def species_stagnant(self, sid, species):
        pass

    def info(self, msg):
        pass


class StdOutReporter(BaseReporter):

    bestFitness = 0.0
    """Uses `print` to output information about the run; an example reporter class."""
    def __init__(self, show_species_detail):
        self.show_species_detail = show_species_detail
        self.generation = None
        self.generation_start_time = None
        self.generation_times = []
        self.num_extinctions = 0

    def start_generation(self, generation):
        info = open("info.txt", "a")
        self.generation = generation
        print('\n ****** Running generation {0} ****** \n'.format(generation))
        info.write('\n ****** Running generation {0} ****** \n\n'.format(generation))
        self.generation_start_time = time.time()
        info.close()

    def end_generation(self, config, population, species_set):
        info = open("info.txt", "a")

        ng = len(population)
        ns = len(species_set.species)
        if self.show_species_detail:
            print('Population of {0:d} members in {1:d} species:'.format(ng, ns))
            info.write('Population of {0:d} members in {1:d} species:\n'.format(ng, ns))
            sids = list(iterkeys(species_set.species))
            sids.sort()
            print("   ID   age  size  fitness  adj fit  stag")
            print("  ====  ===  ====  =======  =======  ====")
            info.write("   ID   age  size  fitness  adj fit  stag\n")
            info.write("  ====  ===  ====  =======  =======  ====\n")
            for sid in sids:
                s = species_set.species[sid]
                a = self.generation - s.created
                n = len(s.members)
                f = "--" if s.fitness is None else "{:.1f}".format(s.fitness)
                af = "--" if s.adjusted_fitness is None else "{:.3f}".format(s.adjusted_fitness)
                st = self.generation - s.last_improved
                print(
                    "  {: >4}  {: >3}  {: >4}  {: >7}  {: >7}  {: >4}".format(sid, a, n, f, af, st))

                info.write(
                    "  {: >4}  {: >3}  {: >4}  {: >7}  {: >7}  {: >4}\n".format(sid, a, n, f, af, st))
                best = None
                for g in itervalues(s.members):
                    if best is None or g.fitness > best.fitness:
                        best = g
                if best.fitness > 10:
                    print(
                        f"Best fitness in specie {s.key} is {best.fitness}, {best.nodes_every_layer}, {list(best.direct_conn.keys())}, {best.cnn_nodes_conv_layer}.\n")
                    info.write(f"Best fitness in specie {s.key} is {best.fitness}, {best.nodes_every_layer}, {list(best.direct_conn.keys())}, {best.cnn_nodes_conv_layer}.\n")

        else:
            print('Population of {0:d} members in {1:d} species'.format(ng, ns))
            info.write('Population of {0:d} members in {1:d} species\n'.format(ng, ns))

        elapsed = time.time() - self.generation_start_time
        self.generation_times.append(elapsed)
        self.generation_times = self.generation_times[-10:]
        average = sum(self.generation_times) / len(self.generation_times)
        print('Total extinctions: {0:d}'.format(self.num_extinctions))
        info.write('Total extinctions: {0:d}\n'.format(self.num_extinctions))
        if len(self.generation_times) > 1:
            print("Generation time: {0:.3f} sec ({1:.3f} average)".format(elapsed, average))
            info.write("Generation time: {0:.3f} sec ({1:.3f} average)\n".format(elapsed, average))
        else:
            print("Generation time: {0:.3f} sec".format(elapsed))
            info.write("Generation time: {0:.3f} sec\n".format(elapsed))

        info.close()

    def post_evaluate(self, config, population, species, best_genome):
        info = open("info.txt", "a")
        # pylint: disable=no-self-use
        fitnesses = [c.fitness for c in itervalues(population)]
        fit_mean = mean(fitnesses)
        fit_std = stdev(fitnesses)
        best_species_id = species.get_species_id(best_genome.key)
        print('Population\'s average fitness: {0:3.5f} stdev: {1:3.5f}'.format(fit_mean, fit_std))
        print(
            'Best fitness: {0:3.5f} - size: {1!r} - species {2} - id {3}'.format(best_genome.fitness,
                                                                                 best_genome.size(),
                                                                                 best_species_id,
                                                                                 best_genome.key))
        print(best_genome.nodes_every_layer)
        info.write('Population\'s average fitness: {0:3.5f} stdev: {1:3.5f}\n'.format(fit_mean, fit_std))
        info.write(
            'Best fitness: {0:3.5f} - size: {1!r} - species {2} - id {3}\n'.format(best_genome.fitness,
                                                                                 best_genome.size(),
                                                                                 best_species_id,
                                                                                 best_genome.key))
        info.write(str(best_genome.nodes_every_layer) + '\n')
        info.close()
        """
        # andrew add
        if (best_genome.fitness > self.bestFitness):
            self.bestFitness = best_genome.fitness
            best = open("best.txt", "a")
            best.write('\nBest genome:\n{!s}'.format(best_genome))
            best.close()
        # andrew end
        """

    def complete_extinction(self):
        self.num_extinctions += 1
        print('All species extinct.')

    def found_solution(self, config, generation, best):
        print('\nBest individual in generation {0} meets fitness threshold - complexity: {1!r}'.format(
            self.generation, best.size()))

    def species_stagnant(self, sid, species):
        if self.show_species_detail:
            print("\nSpecies {0} with {1} members is stagnated: removing it".format(sid, len(species.members)))

    def info(self, msg):
        print(msg)
