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

def eval_fitness(output, target, genome_id):
    info = open("info.txt", "a")
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
    print(f"-------Accuracy of Genome {genome_id} is {total_matched_rows}/640----------")
    info.write(f"-------Accuracy of Genome {genome_id} is {total_matched_rows}/640----------\n")
    info.close()
    return fitness.item()*10+200
    # return total_matched_rows/10
    # return (fitness.item() * 10 + 200)/2 + total_matched_rows/15

def eval_genomes(genomes, config, input_, target, gso):
    if torch.cuda.is_available():
        input = input_[0].cuda()
        input_p = input_[1].cuda()
        target = target.cuda()
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = evaluate_torch.Net(config, genome)
        if torch.cuda.is_available():
            net.cuda()
            for layer in net.cnn_layers:
                for node in layer:
                    node = node.cuda()
                    for param in node.parameters():
                        param.data = param.data.cuda()
        batch_output_allAgent = net(input, input_p, gso)
        genome.fitness += eval_fitness(batch_output_allAgent, target, genome_id)

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--model_type', type=str, default='il')

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config/config_neat')
    pretrained_fc = os.path.join(local_dir, 'save.pkl')

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    dataloader = DecentralPlannerDataLoader(config_train)

    networks = neat.Population(config, pretrained_fc, dataloader)
    networks.add_reporter(neat.StdOutReporter(True))
    winner = networks.run(eval_genomes)


if __name__ == '__main__':
    main()
