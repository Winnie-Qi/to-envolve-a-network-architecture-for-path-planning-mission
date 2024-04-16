'''
@Author: Weijie Qi
@Date: 2024
'''
import os
import argparse
import torch
import torch.nn as nn
import neat
import evaluate.evaluate_torch as evaluate_torch

from config.config_train import config_train
from dataloader.decentral_dataloader import DecentralPlannerDataLoader

def eval_fitness(output, target):
    fitness = 0
    total_matched_rows = 0
    target = target.permute(1, 0, 2)
    criterion = nn.CrossEntropyLoss()
    for agent in range(len(output)):
        fitness += -criterion(output[agent], torch.max(target[agent], 1)[1])
        max_indices = torch.argmax(output[agent], dim=1)
        one_indices = torch.nonzero(target[agent], as_tuple=False)
        matched_indices = torch.eq(max_indices, one_indices[:, 1])
        total_matched_rows += torch.sum(matched_indices).item()
    print(f"-------Accuracy is {total_matched_rows}/640----------")
    return fitness.item()

def eval_genomes(genomes, config, input, target, gso):
    if torch.cuda.is_available():
        input = input.cuda()
        target = target.cuda()
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = evaluate_torch.Net(config, genome)
        if torch.cuda.is_available():
            net.cuda()
        batch_output_allAgent = net(input, gso)
        genome.fitness += eval_fitness(batch_output_allAgent, target)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--model_type', type=str, default='il')

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config/config_neat')

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    dataloader = DecentralPlannerDataLoader(config_train)

    networks = neat.Population(config, dataloader)
    networks.add_reporter(neat.StdOutReporter(True))
    winner = networks.run(eval_genomes)


if __name__ == '__main__':
    main()
