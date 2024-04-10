
class Config(object):
    def __init__(self):
        self.AgentNum = 10
        self.map_w = 20
        self.map_h = 20
        self.max_epoch = 0 # 150
        self.model_type = 'imitation_learning'
        self.learning_rate = 0.001
        self.weight_decay = 0.00001
        self.validate_every = 5
        self.data_root = 'C:/Projects/gnn_pathplanning-master/data/DataSource_Dmap'
        self.num_test_trainingSet = 500
        self.num_validset = 200
        self.batch_size = 64
        self.data_loader_workers = 4
        self.pin_memory = True
        self.valid_batch_size = 1
        self.log_interval = 500
        self.max_step = 100

config_train = Config()
