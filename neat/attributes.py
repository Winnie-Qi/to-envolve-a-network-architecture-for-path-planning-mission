"""Deals with the attributes (variable parameters) of genes"""
from random import choice, gauss, random, uniform
from neat.config import ConfigParameter
from neat.six_util import iterkeys, iteritems


# TODO: There is probably a lot of room for simplification of these classes using metaprogramming.


class BaseAttribute(object):
    """Superclass for the type-specialized attribute subclasses, used by genes."""

    def __init__(self, name, **default_dict):
        self.name = name
        for n, default in iteritems(default_dict):
            self._config_items[n] = [self._config_items[n][0], default]
        for n in iterkeys(self._config_items):
            setattr(self, n + "_name", self.config_item_name(n))

    def config_item_name(self, config_item_base_name):
        return "{0}_{1}".format(self.name, config_item_base_name)

    def get_config_params(self):
        return [ConfigParameter(self.config_item_name(n),
                                self._config_items[n][0],
                                self._config_items[n][1])
                for n in iterkeys(self._config_items)]


class FloatAttribute(BaseAttribute):
    """
    Class for numeric attributes,
    such as the response of a node or the weight of a connection.
    """
    _config_items = {"init_mean": [float, None],
                     "init_stdev": [float, None],
                     "init_type": [str, 'gaussian'],
                     "replace_rate": [float, None],
                     "mutate_rate": [float, None],
                     "mutate_power": [float, None],
                     "max_value": [float, None],
                     "min_value": [float, None]}

    def clamp(self, value, config):
        min_value = getattr(config, self.min_value_name)
        max_value = getattr(config, self.max_value_name)
        return max(min(value, max_value), min_value)

    def init_value(self, config, num_nodes):
        mean = getattr(config, self.init_mean_name)
        stdev = getattr(config, self.init_stdev_name)
        if self.init_stdev_name == 'weight_init_stdev':
            stdev = (2 / (num_nodes[0] + num_nodes[1])) ** 0.5
        init_type = getattr(config, self.init_type_name).lower()

        if ('gauss' in init_type) or ('normal' in init_type):
            return self.clamp(gauss(mean, stdev), config)

        if 'uniform' in init_type:
            min_value = max(getattr(config, self.min_value_name),
                            (mean - (2 * stdev)))
            max_value = min(getattr(config, self.max_value_name),
                            (mean + (2 * stdev)))
            return uniform(min_value, max_value)

        raise RuntimeError("Unknown init_type {!r} for {!s}".format(getattr(config,
                                                                    self.init_type_name),
                                                                    self.init_type_name))

    def mutate_value(self, value, config, num_nodes):
        # mutate_rate is usually no lower than replace_rate, and frequently higher -
        # so put first for efficiency
        mutate_rate = getattr(config, self.mutate_rate_name)

        r = random()
        if r < mutate_rate:
            mutate_power = getattr(config, self.mutate_power_name)
            return self.clamp(value + gauss(0.0, mutate_power), config)

        replace_rate = getattr(config, self.replace_rate_name)

        if r < replace_rate + mutate_rate:
            return self.init_value(config, num_nodes)

        return value

    def validate(self, config):  # pragma: no cover
        pass

class IntAttribute(BaseAttribute):
    _config_items = {}
    def init_value(self, config, _):
        return False

class BoolAttribute(BaseAttribute):
    """Class for boolean attributes such as whether a connection is enabled or not."""
    _config_items = {"default": [bool, False]}

    def init_value(self, config, _):
        return False

    def mutate_value(self, value, config, _):
        pass

    def validate(self, config):  # pragma: no cover
        pass

class ListAttribute(BaseAttribute):
    """
    Class for numeric attributes,
    such as the response of a node or the weight of a connection.
    """
    _config_items = {"init_mean": [float, None],
                     "init_stdev": [float, None],
                     "init_type": [str, 'gauss'],
                     "replace_rate": [float, None],
                     "mutate_rate": [float, None],
                     "mutate_power": [float, None],
                     "max_value": [float, None],
                     "min_value": [float, None],
                     "size":[int, None]}

    def clamp(self, value, config):
        min_value = getattr(config, self.min_value_name)
        max_value = getattr(config, self.max_value_name)
        return max(min(value, max_value), min_value)

    def init_value(self, config, num_nodes):
        value = list()

        mean = getattr(config, self.init_mean_name)
        stdev = (2/(config.kernel_size * num_nodes[0] + config.kernel_size * num_nodes[1]))**0.5
        init_type = getattr(config, self.init_type_name).lower()

        for i in range(config.kernel_size * num_nodes[0]):
            if ('gauss' in init_type) or ('normal' in init_type):
                value.append(self.clamp(gauss(mean, stdev), config))
            elif 'uniform' in init_type:
                min_value = max(getattr(config, self.min_value_name),
                                (mean - (2 * stdev)))
                max_value = min(getattr(config, self.max_value_name),
                                (mean + (2 * stdev)))
                value.append(uniform(min_value, max_value))
            else:
                raise RuntimeError("Unknown init_type {!r} for {!s}".format(getattr(config,
                                                                        self.init_type_name),
                                                                        self.init_type_name))
        return value

    def init_single_value(self, config, num_nodes):

        mean = getattr(config, self.init_mean_name)
        stdev = (2/(config.kernel_size * num_nodes[0] + config.kernel_size * num_nodes[1]))**0.5
        init_type = getattr(config, self.init_type_name).lower()

        if ('gauss' in init_type) or ('normal' in init_type):
            return self.clamp(gauss(mean, stdev), config)

        if 'uniform' in init_type:
            min_value = max(getattr(config, self.min_value_name),
                            (mean - (2 * stdev)))
            max_value = min(getattr(config, self.max_value_name),
                            (mean + (2 * stdev)))
            return uniform(min_value, max_value)

        raise RuntimeError("Unknown init_type {!r} for {!s}".format(getattr(config,
                                                                    self.init_type_name),
                                                                    self.init_type_name))

    def mutate_value(self, value, config, num_nodes):
        # mutate_rate is usually no lower than replace_rate, and frequently higher -
        # so put first for efficiency
        mutate_rate = getattr(config, self.mutate_rate_name)
        replace_rate = getattr(config, self.replace_rate_name)
        mutate_power = getattr(config, self.mutate_power_name)

        for i in range(len(value)):
            r = random()
            if r < mutate_rate:
                value[i] = self.clamp(value[i] + gauss(0.0, mutate_power), config)
            elif r < replace_rate + mutate_rate:
                value[i] =  self.init_single_value(config, num_nodes)

        return value

    def add_layer(self, config, num_nodes):
        value = list()

        mean = getattr(config, self.init_mean_name)
        stdev = (2 / (config.kernel_size * num_nodes[0] + config.kernel_size * num_nodes[1])) ** 0.5

        for i in range(config.kernel_size):
            value.append(self.clamp(gauss(mean, stdev), config))

        return value