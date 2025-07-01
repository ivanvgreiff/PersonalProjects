import jax
import optax
from ml_collections import config_dict

import Models

#* Configuration
config = config_dict.ConfigDict()
config.seed = 1 #! Not used currently
config.experiment = 0 #! Not used currently

#* Training Configuration
config.train = config_dict.ConfigDict()
config.train.num_runs = 1
config.train.model = Models.lenet5
config.train.loss_fn = optax.softmax_cross_entropy
# config.train.use_reg = True
# config.train.weight_decay = 1e-5
config.train.batchwise_update = True
# config.train.metric = config_dict.ConfigDict()
# config.train.metric.loss = metrics.SparseCategoricalCrossentropy
# config.train.metric.acc = metrics.SparseCategoricalAccuracy

#* Data Configuration
config.data = config_dict.ConfigDict()
config.data.name = 'mnist'
config.data.path = "./data/"
config.data.alloc_type = 'uniform'
config.data.alloc_ratio = 2
config.data.beta = 0.5
config.data.batch_size = 512
config.data.num_validation = 100
config.data.shape = () #! TBD in Code 
config.data.num_classes = 0 #! TBD in Code 

config.beta = config_dict.ConfigDict()
config.beta.mode = 'value' 
config.beta.value = 0.5
config.beta.index = 0

#*Server Configuration 
config.server = config_dict.ConfigDict()

config.server.num_epochs = 50
config.server.update_thresh = 0.75
config.server.es = config_dict.ConfigDict()
config.server.es.enable = False
config.server.es.threshold = 0.1 #! Not used currently
config.server.es.wait = 0
config.server.es.patience = 3

config.server.tx = config_dict.ConfigDict()
config.server.tx.fn = optax.sgd
config.server.tx.lr = 1

#*Worker Configuration
config.worker = config_dict.ConfigDict()
config.worker.num = 10
config.worker.inact_prob = 0

config.worker.epoch = config_dict.ConfigDict()
config.worker.epoch.type = 'constant'
config.worker.epoch.mean = 3
config.worker.epoch.std = 2
config.worker.epoch.beta = 0.5
config.worker.epoch.coef = 20
config.worker.epoch.is_random = False

config.worker.tx = config_dict.ConfigDict()
config.worker.tx.fn = optax.sgd
config.worker.tx.lr = 0.01
# config.worker.tx.lr_decay_per_server_epoch = 1.0
# config.worker.tx.type_decay_per_server_epoch = 'Geometric'
# config.worker.tx.lr_decay_per_worker_epoch = 1.0
# config.worker.tx.type_decay_per_worker_epoch = 'Geometric'
config.worker.tx.moment = 0.9


config.data.poison = config_dict.ConfigDict()
config.data.poison.patch = config_dict.ConfigDict()
config.data.poison.patch.size = (7, 7, 1)
config.data.poison.patch.loc = 0
config.data.poison.patch.loc_rad = 2.5
config.data.poison.patch.val = 1.0
config.data.poison.calc_l2_norm = True
config.data.poison.source_clss = 5
config.data.poison.target_clss = 6
config.data.poison.ratio = 0.2
config.data.poison.clients = [0, 9]

config.quantization = config_dict.ConfigDict()
config.quantization.prime = 127
config.quantization.levels = 1e3

config.rlr = config_dict.ConfigDict()
config.rlr.thresh = 8 


# #* Compressor Configuration
# config.compressor = config_dict.ConfigDict()
# config.compressor.enable = False

# config.compressor.s2w = config_dict.ConfigDict()
# config.compressor.s2w.comp = Compress.TopKCompressor
# config.compressor.s2w.enable = False
# config.compressor.s2w.error_fb = None
# config.compressor.s2w.clipping = None
# config.compressor.s2w.rescale = True
# config.compressor.s2w.k = 1000
# config.compressor.s2w.prob = 0.0 #! TBD in Code

# config.compressor.w2s = config_dict.ConfigDict()
# config.compressor.w2s.comp = Compress.TopKCompressor
# config.compressor.w2s.enable = True
# config.compressor.w2s.error_fb = None
# config.compressor.w2s.clipping = None
# config.compressor.w2s.rescale = True
# config.compressor.w2s.k = 1000
# config.compressor.w2s.prob = 0.0 #! TBD in Code
# config.compressor.w2s.beta = 0.9

# #*Distilllation Configuration
# config.distillation = config_dict.ConfigDict()
# config.distillation.num_epochs = 5

# config.distillation.syn = config_dict.ConfigDict()
# config.distillation.syn.lr = config_dict.ConfigDict()
# config.distillation.syn.lr.minval = 0.01
# config.distillation.syn.lr.maxval = 0.1

# config.distillation.syn.data = config_dict.ConfigDict()
# config.distillation.syn.data.num = 50
# config.distillation.syn.data.mean = 5
# config.distillation.syn.data.std = 1
# config.distillation.syn.data.label_dominance = 100
# config.distillation.syn.data.label_as_prob = False
# config.distillation.syn.data.batch_size = 5

# config.distillation.tx = config_dict.ConfigDict()
# config.distillation.tx.name = optax.sgd.__name__ #! Must find another way to make this work 
# config.distillation.tx.lr = 0.02

config.wandb = config_dict.ConfigDict()
config.wandb.name = '' #! TBD in Code
config.wandb.project = 'Federated Learning'
config.wandb.job_type = 'Federated Learning'