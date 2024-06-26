"""Handles node and connection genes."""
import warnings
import copy
from random import random
from neat.attributes import FloatAttribute, IntAttribute, BoolAttribute, ListAttribute

# TODO: There is probably a lot of room for simplification of these classes using metaprogramming.
# TODO: Evaluate using __slots__ for performance/memory usage improvement.


class BaseGene(object):
    """
    Handles functions shared by multiple types of genes (both node and connection),
    including crossover and calling mutation methods.
    """
    def __init__(self, key):
        self.key = key

    def __str__(self):
        attrib = ['key'] + [a.name for a in self._gene_attributes]
        attrib = ['{0}={1}'.format(a, getattr(self, a)) for a in attrib]
        return '{0}({1})'.format(self.__class__.__name__, ", ".join(attrib))

    def __lt__(self, other):
        assert isinstance(self.key,type(other.key)), "Cannot compare keys {0!r} and {1!r}".format(self.key,other.key)
        return self.key < other.key

    @classmethod
    def parse_config(cls, config, param_dict):
        pass

    @classmethod
    def get_config_params(cls):
        params = []
        if not hasattr(cls, '_gene_attributes'):
            setattr(cls, '_gene_attributes', getattr(cls, '__gene_attributes__'))
            warnings.warn(
                "Class '{!s}' {!r} needs '_gene_attributes' not '__gene_attributes__'".format(
                    cls.__name__,cls),
                DeprecationWarning)
        for a in cls._gene_attributes:
            params += a.get_config_params()
        return params

    def init_attributes(self, config, num_nodes):
        for a in self._gene_attributes:
            setattr(self, a.name, a.init_value(config, num_nodes))

    def mutate(self, config, num_nodes):
        for a in self._gene_attributes:
            if a.name == 'single_channel':
                continue
            v = getattr(self, a.name)
            setattr(self, a.name, copy.deepcopy(a.mutate_value(v, config, num_nodes)))

    def copy(self):
        if isinstance(self, DefaultNodeGeneFC):
            new_gene = self.__class__(self.key, self.layer)
        elif isinstance(self, DefaultNodeGeneCNN):
            new_gene = self.__class__(self.key, self.layer)
        else: # is a connection
            new_gene = self.__class__(self.key, self.connect_layer)
        for a in self._gene_attributes:
            setattr(new_gene, a.name, copy.deepcopy(getattr(self, a.name)))

        return new_gene

    def crossover(self, gene2):
        """ Creates a new gene randomly inheriting attributes from its parents."""
        assert self.key == gene2.key

        if isinstance(self, DefaultNodeGeneFC): # is a fc node
            assert self.layer == gene2.layer
            new_gene = self.__class__(self.key, self.layer)
        elif isinstance(self, DefaultNodeGeneCNN):  # is a cnn node
            assert self.layer == gene2.layer
            new_gene = self.__class__(self.key, self.layer)
        else: # is a connection
            new_gene = self.__class__(self.key, self.connect_layer)

        # Note: we use "a if random() > 0.5 else b" instead of choice((a, b))
        # here because `choice` is substantially slower.
        for a in self._gene_attributes:
            if a.name != 'kernel':
                if random() < 0.6:  # Inherit the first gene
                    setattr(new_gene, a.name, copy.deepcopy(getattr(self, a.name)))
                else: # Inherit the second gene
                    setattr(new_gene, a.name, copy.deepcopy(getattr(gene2, a.name)))
            else:
                if random() < 0.4 and len(getattr(self, a.name)) == len(getattr(gene2, a.name)):  # Inherit the first gene
                    setattr(new_gene, a.name, copy.deepcopy(getattr(gene2, a.name)))
                else: # Inherit the second gene
                    setattr(new_gene, a.name, copy.deepcopy(getattr(self, a.name)))

# @@@@@@@@@@ andrew begin

            if (random() < 0.5):
                if ((a.name == 'weight') or (a.name == 'bias')):
                    lamda = random()
                    tmpa = getattr(self, a.name)
                    tmpb = getattr(gene2, a.name)
                    tmp = tmpa * lamda + tmpb * (1 - lamda)
                    setattr(new_gene, a.name, tmp)
                elif (a.name == 'kernel'):
                    for i in range(len(new_gene.kernel)):
                        lamda = random()
                        try:
                            tmpa = self.kernel[i]
                        except:
                            tmpa = sum(self.kernel) / len(self.kernel)
                        try:
                            tmpb = gene2.kernel[i]
                        except:
                            tmpb = sum(gene2.kernel) / len(gene2.kernel)
                        tmp = tmpa * lamda + tmpb * (1 - lamda)
                        new_gene.kernel[i] = tmp
# @@@@@@@@@@ andrew end

        return new_gene


# TODO: Should these be in the nn module?  iznn and ctrnn can have additional attributes.


class DefaultNodeGeneCNN(BaseGene):
    _gene_attributes = [FloatAttribute('bias'),
                        IntAttribute('single_channel'),
                        BoolAttribute('cancle_padding'),
                        ListAttribute('kernel')]

    def __init__(self, key, layer):
        assert isinstance(key, int), "DefaultNodeGene key must be an int, not {!r}".format(key)
        BaseGene.__init__(self, key)
        self.layer = layer

    def distance(self, other, config):
        d = abs(self.bias - other.bias)
        d += sum([abs(x - y) for x, y in zip(self.kernel, other.kernel)])/ (10 * len(self.kernel))
        if self.single_channel != other.single_channel:
            d += 100.0
        if self.cancle_padding != other.cancle_padding:
            d += 1.0
        return d * config.compatibility_weight_coefficient

class DefaultNodeGeneFC(BaseGene):
    _gene_attributes = [FloatAttribute('bias')]

    def __init__(self, key, layer):
        assert isinstance(key, int), "DefaultNodeGene key must be an int, not {!r}".format(key)
        BaseGene.__init__(self, key)
        self.layer = layer

    def distance(self, other, config):
        d = abs(self.bias - other.bias)
        return d * config.compatibility_weight_coefficient

# TODO: Do an ablation study to determine whether the enabled setting is
# important--presumably mutations that set the weight to near zero could
# provide a similar effect depending on the weight range, mutation rate,
# and aggregation function. (Most obviously, a near-zero weight for the
# `product` aggregation function is rather more important than one giving
# an output of 1 from the connection, for instance!)
class DefaultConnectionGene(BaseGene):
    _gene_attributes = [FloatAttribute('weight')]

    def __init__(self, key, i):
        assert isinstance(key, tuple), "DefaultConnectionGene key must be a tuple, not {!r}".format(key)
        self.connect_layer = [i[0],i[1]]
        BaseGene.__init__(self, key)

    def distance(self, other, config):
        d = abs(self.weight - other.weight)
        return d * config.compatibility_weight_coefficient
