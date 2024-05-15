"""Divides the population into species based on genomic distances."""
from itertools import count
from random import random

from neat.math_util import mean, stdev
from neat.six_util import iteritems, iterkeys, itervalues
from neat.config import ConfigParameter, DefaultClassConfig

class Species(object):
    def __init__(self, key, generation):
        self.key = key
        self.created = generation
        self.last_improved = generation
        self.representative = None
        self.members = {}
        self.fitness = None
        self.adjusted_fitness = None
        self.fitness_history = []

    def update(self, representative, members):
        self.representative = representative
        self.members = members

    def get_fitnesses(self):
        return [m.fitness for m in itervalues(self.members)]


class GenomeDistanceCache(object):
    def __init__(self, config):
        self.distances = {}
        self.config = config
        self.hits = 0
        self.misses = 0

    def __call__(self, genome0, genome1):
        g0 = genome0.key
        g1 = genome1.key
        if genome0.num_cnn_layer != genome1.num_cnn_layer or genome0.dense_after_cnn != genome1.dense_after_cnn or genome0.dense_after_gnn != genome1.dense_after_gnn:
            return 100
        if any([abs(x - y) > 6 for x, y in zip(genome0.nodes_every_layer, genome1.nodes_every_layer)]):
            return 100
        if abs(sum(genome0.nodes_every_layer) - sum(genome1.nodes_every_layer)) > 12:
            return 100
        if abs(len(genome0.direct_conn) - len(genome1.direct_conn)) > 15:
            return 100
        d = 0
        for k1, n1 in iteritems(genome0.nodes):
            if hasattr(n1, 'kernel'):
                n2 = genome1.nodes.get(k1)
                if n2 is not None:
                    if n1.single_channel != n2.single_channel:
                        d += 1
                    if n1.cancle_padding != n2.cancle_padding:
                        d += 1
        # s = 0
        # for i in range(-1, 3):
        #     s += abs(genome0.cnn_nodes_conv_layer[0].count(i) - genome1.cnn_nodes_conv_layer[0].count(i))
        #     if s >= 0.25 * len(genome0.cnn_nodes_conv_layer[0]):
        #         return 100
        # d = self.distances.get((g0, g1))
        # if d is None:
            # Distance is not already computed.
        d = genome0.distance(genome1, self.config)
        self.distances[g0, g1] = d
        self.distances[g1, g0] = d
        self.misses += 1
        # else:
        #     self.hits += 1

        return d

class DefaultSpeciesSet(DefaultClassConfig):
    """ Encapsulates the default speciation scheme. """

    def __init__(self, config, reporters):
        # pylint: disable=super-init-not-called
        self.species_set_config = config
        self.reporters = reporters
        self.indexer = count(1)
        self.species = {}
        self.genome_to_species = {}

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('compatibility_threshold', float)])

    def speciate(self, config, population, generation):
        """
        Place genomes into species by genetic similarity.

        Note that this method assumes the current representatives of the species are from the old
        generation, and that after speciation has been performed, the old representatives should be
        dropped and replaced with representatives from the new generation.  If you violate this
        assumption, you should make sure other necessary parts of the code are updated to reflect
        the new behavior.
        """
        assert isinstance(population, dict)

        compatibility_threshold = self.species_set_config.compatibility_threshold

        # Find the best representatives for each existing species.
        unspeciated = set(iterkeys(population))
        distances = GenomeDistanceCache(config.genome_config)
        new_representatives = {}
        new_members = {}
        for sid, s in iteritems(self.species):
            candidates = []
            for gid in unspeciated:
                g = population[gid]
                d = distances(s.representative, g)
                candidates.append((d, g))

            # The new representative is the genome closest to the current representative.
            ignored_rdist, new_rep = min(candidates, key=lambda x: x[0])
            new_rid = new_rep.key
            new_representatives[sid] = new_rid
            new_members[sid] = [new_rid]
            unspeciated.remove(new_rid)

        # Merge Species if evolve into a similar topology
        delete_specie = set()
        for sid, s in iteritems(self.species):
            for gid, g in iteritems(self.species):
                if sid != gid and sid not in delete_specie and gid not in delete_specie:
                    d = distances(s.representative, g.representative)
                    if d < compatibility_threshold:
                        if sid > gid:
                            sid, gid = gid, sid # make sure sid < gid
                        delete_specie.add(gid)
                        new_members[sid].extend(new_members[gid])
                        del new_members[gid]
                        del new_representatives[gid]
                        print(f'Delete specie {gid} because it evolved to resemble existing species. ')
        for g in delete_specie:
            self.species.pop(g)

        # Partition population into species based on genetic similarity.
        while unspeciated:
            gid = unspeciated.pop()
            g = population[gid]

            # Find the species with the most similar representative.
            candidates = []
            for sid, rid in iteritems(new_representatives):
                rep = population[rid]
                d = distances(rep, g)
                if d < compatibility_threshold:
                    candidates.append((d, sid))

            if candidates:
                ignored_sdist, sid = min(candidates, key=lambda x: x[0])
                new_members[sid].append(gid)
            else:
                # No species is similar enough, create a new species, using
                # this genome as its representative.
                sid = next(self.indexer)
                new_representatives[sid] = gid
                new_members[sid] = [gid]

        # Update species collection based on new speciation.
        self.genome_to_species = {}
        for sid, rid in iteritems(new_representatives):
            s = self.species.get(sid)
            if s is None:
                s = Species(sid, generation)
                self.species[sid] = s

            members = new_members[sid]
            for gid in members:
                self.genome_to_species[gid] = sid

            member_dict = dict((gid, population[gid]) for gid in members)
            s.update(population[rid], member_dict)

        gdmean = mean(itervalues(distances.distances))
        gdstdev = stdev(itervalues(distances.distances))
        self.reporters.info(
            'Mean genetic distance {0:.3f}, standard deviation {1:.3f}'.format(gdmean, gdstdev))

    def get_species_id(self, individual_id):
        return self.genome_to_species[individual_id]

    def get_species(self, individual_id):
        sid = self.genome_to_species[individual_id]
        return self.species[sid]
