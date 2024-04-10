"""
An example for dataset loaders, starting with data loading including all the functions that either preprocess or postprocess data.
"""

import os
import random
import torch
import numpy as np
import logging
import scipy.io as sio
from torch.utils.data import DataLoader
from torch.utils import data

from .statetransformer import AgentState


class DecentralPlannerDataLoader:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("DecentralPlannerDataLoader")
        log_info = "Loading #{} Agent DATA from dataset .....".format(self.config.AgentNum)
        self.logger.info(log_info)

        train_set = CreateDataset(self.config, "train")
        valid_set = CreateDataset(self.config, "valid")
        test_set = CreateDataset(self.config, "test")

        self.train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True,
                                       num_workers=self.config.data_loader_workers,
                                       pin_memory=self.config.pin_memory)

        self.valid_loader = DataLoader(valid_set, batch_size=self.config.batch_size, shuffle=True,
                                       num_workers=self.config.data_loader_workers,
                                       pin_memory=self.config.pin_memory)

        self.test_loader = DataLoader(test_set, batch_size=self.config.valid_batch_size, shuffle=True,
                                       num_workers=self.config.data_loader_workers,
                                       pin_memory=self.config.pin_memory)

    def finalize(self):
        pass


class CreateDataset(data.Dataset):
    def __init__(self, config, mode):
        self.config = config
        self.datapath_exp = 'map{:02d}x{:02d}/{}_agent/'.format(self.config.map_w, self.config.map_h,self.config.AgentNum)
        self.dirName = os.path.join(self.config.data_root, self.datapath_exp)
        self.AgentState = AgentState(self.config.AgentNum)

        if mode == "train":
            self.dir_data = os.path.join(self.dirName, 'train')
            self.search_files = self.search_target_files_withStep
            self.data_paths, self.id_stepdata = self.update_data_path_trainingset(self.dir_data)
            self.load_data = self.load_train_data

        elif mode == 'valid':
            self.dir_data = os.path.join(self.dirName, 'valid')
            self.search_files = self.search_target_files_withStep
            self.data_paths, self.id_stepdata = self.update_data_path_trainingset(self.dir_data)
            self.load_data = self.load_train_data

        elif mode == "test":
            self.dir_data = os.path.join(self.dirName, 'valid')
            self.data_paths, self.id_stepdata = self.obtain_data_path_validset(self.dir_data, self.config.num_validset)
            self.load_data = self.load_data_during_training

        self.data_size = len(self.data_paths)

    def __getitem__(self, index):

        path = self.data_paths[index % self.data_size]
        id_step = int(self.id_stepdata[index % self.data_size])
        input, target, GSO, map_tensor = self.load_data(path, id_step)
        return input, target, id_step, GSO, map_tensor

    def update_data_path_trainingset(self, dir_data):
        # only used for training set and online expert - training purpose
        data_paths_total = []
        step_paths_total = []
        # load common training set (21000)
        data_paths, step_paths = self.search_files(dir_data)
        data_paths_total.extend(data_paths)
        step_paths_total.extend(step_paths)
        # load training set from online expert based on failCases
        # data_paths_failcases, step_paths_failcases = self.search_files(self.config.failCases_dir)
        # data_paths_total.extend(data_paths_failcases)
        # step_paths_total.extend(step_paths_failcases)
        paths_total = list(zip(data_paths_total, step_paths_total))
        random.shuffle(paths_total)
        data_paths_total, step_paths_total = zip(*paths_total)
        return data_paths_total, step_paths_total

    def obtain_data_path_validset(self, dir_data, case_limit):
        # obtain validation data to valid the decision making at given state
        data_paths, id_stepdata = self.search_target_files(dir_data)
        paths_bundle = list(zip(data_paths, id_stepdata))
        paths_bundle = sorted(paths_bundle)
        data_paths, id_stepdata = zip(*paths_bundle)
        data_paths = data_paths[:case_limit]
        id_stepdata = id_stepdata[:case_limit]
        return data_paths, id_stepdata

    def load_train_data(self, path, id_step):
        input = []

        data_contents = sio.loadmat(path)
        map_channel = data_contents['map']  # W x H

        input_tensor = data_contents['inputTensor'] # step x num_agent x 3 x 11 x 11
        # start_location = data_contents['inputState'][0] #
        target_sequence = data_contents['target'] # step x num_agent x 5
        # goal_location = data_contents['goal'] #
        input_GSO_sequence = data_contents['GSO'] # Step x num_agent x num_agent

        tensor_map = torch.from_numpy(map_channel).float()
        inputState = data_contents['inputState']

        step_input_tensor = torch.from_numpy(input_tensor[id_step][:]).float()
        step_input_GSO = torch.from_numpy(input_GSO_sequence[id_step, :, :]).float()
        step_target = torch.from_numpy(target_sequence[id_step, :, :]).long()
        step_inputState = torch.from_numpy(inputState[id_step][:]).int() # (10,2)
        input.append(step_input_tensor)

        goal = data_contents['inputState'][-1]
        goal_state = np.stack((goal, step_inputState))
        goal_state = torch.FloatTensor(goal_state)
        input.append(goal_state)

        return input, step_target, step_input_GSO, tensor_map


    def load_data_during_training(self, path, _):
        # load dataset into validation mode during training - only initial position, predict action towards goal
        # test on training set and test on validation set
        data_contents = sio.loadmat(path)
        map_channel = data_contents['map'] # W x H
        goal_allagents = data_contents['goal'] # num_agent x 2

        input_sequence = data_contents['inputState'][0] # from step x num_agent x 2 to # initial pos x num_agent x 2
        target_sequence = data_contents['target'] # step x num_agent x 5

        self.AgentState.setmap(map_channel)
        step_input_tensor = self.AgentState.stackinfo(goal_allagents, input_sequence)

        step_target = torch.from_numpy(target_sequence).long()
        # from step x num_agent x action (5) to  id_agent x step x action(5)
        step_target = step_target.permute(1, 0, 2)
        step_input_rs = step_input_tensor.squeeze(0)
        step_target_rs = step_target.squeeze(0)

        tensor_map = torch.from_numpy(map_channel).float()
        GSO_none = torch.zeros(1)
        return step_input_rs, step_target_rs, GSO_none, tensor_map


    def load_test_data(self, path, _):
        # load dataset into test mode - only initial position, predict action towards goal

        data_contents = sio.loadmat(path)
        map_channel = data_contents['map'] # W x H
        goal_allagents = data_contents['goal'] # num_agent x 2

        input_sequence = data_contents['inputState']  # num_agent x 2
        target_sequence = data_contents['target']  # step x num_agent x 5

        self.AgentState.setmap(map_channel)
        step_input_tensor = self.AgentState.stackinfo(goal_allagents, input_sequence)

        step_target = torch.from_numpy(target_sequence).long()
        # from step x num_agent x action (5) to  id_agent x step x action(5)
        step_target = step_target.permute(1, 0, 2)
        step_input_rs = step_input_tensor.squeeze(0)
        step_target_rs = step_target.squeeze(0)

        tensor_map = torch.from_numpy(map_channel).float()
        GSO_none = torch.zeros(1)
        return step_input_rs, step_target_rs, GSO_none, tensor_map

    def search_target_files(self, dir):
        # make a list of file name of input yaml
        list_path = []
        list_path_stepdata = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if self.is_target_file(fname):
                    makespan = int(fname.split('_MP')[-1].split('.mat')[0])
                    path = os.path.join(root, fname)
                    list_path.append(path)
                    list_path_stepdata.append(makespan)

        return list_path, list_path_stepdata

    def search_target_files_withStep(self, dir):
        # make a list of file name of input yaml
        list_path = []
        list_path_stepdata = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if self.is_target_file(fname):
                    makespan = int(fname.split('_MP')[-1].split('.mat')[0])
                    path = os.path.join(root, fname)
                    for step in range(makespan):
                        # path = os.path.join(root, fname, str(step))
                        list_path.append(path)
                        # print('*PATH*', path)
                        list_path_stepdata.append(step)
                        # print('*STEP*', step)

        return list_path, list_path_stepdata

    def search_valid_files_withStep(self, dir, case_limit):
        # make a list of file name of input yaml
        list_path = []
        list_path_stepdata = []
        count_num_cases = 0
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if self.is_target_file(fname):
                    makespan = int(fname.split('_MP')[-1].split('.mat')[0])
                    path = os.path.join(root, fname)
                    if count_num_cases <= case_limit:
                        for step in range(makespan):
                            # path = os.path.join(root, fname, str(step))
                            list_path.append(path)
                            list_path_stepdata.append(step)
                        count_num_cases += 1
                    else:
                        break

        return list_path, list_path_stepdata

    def is_target_file(self, filename):
        DATA_EXTENSIONS = ['.mat']
        return any(filename.endswith(extension) for extension in DATA_EXTENSIONS)

    def __len__(self):
        return self.data_size


#
# if __name__ == '__main__':
#     config = {'mode': "train",
#               'num_agents': 8,
#               'map_w': 20,
#               'map_h': 20,
#               'map_density': 1,
#
#               'data_root': '/local/scratch/ql295/Data/Project/OnlineExpert_testbed/Quick_Test/DataSourceTri_nonTF_ECBS',
#               'failCases_dir': '/local/scratch/ql295/Data/Project/OnlineExpert_testbed/Quick_Test/Cache_data',
#               'exp_net': 'dcp',
#               "num_test_trainingSet": 2,
#               "num_validStep": 100,
#               "num_validset": 2,
#               "num_testset": 1,
#
#               "data_loader_workers": 0,
#               "pin_memory": True,
#               "async_loading": True,
#
#               "batch_size": 64,
#               "valid_batch_size": 1,
#               "test_batch_size": 1
#               }
#     config_setup = EasyDict(config)
#     data_loader = DecentralPlannerDataLoader(config_setup)
#
