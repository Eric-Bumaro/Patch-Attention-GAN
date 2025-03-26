import torch.cuda


class TrainConfig:
    def __init__(self):
        self.result_dir = "train_result"
        self.experiment_name = "baseline_night2day"
        self.pretrain_weight_G_A = None
        self.pretrain_weight_G_B = None
        self.pretrain_weight_D_A = None
        self.pretrain_weight_D_B = None
        self.save_epoch_freq = 20
        self.input_nc = 3
        self.output_nc = 3
        self.lr = 0.0004
        self.init_gain = 0.02
        self.pool_size = 0
        self.n_epochs = 40
        self.epoch = 0
        self.decay_epoch = 20
        self.warm_up_epoch = 10
        self.ndf = 32
        self.gpu_ids = [0] if torch.cuda.is_available() else [-1]
        self.gan_mode = "lsgan"  # 'vanilla', 'wgangp', "lsgan"
        self.init_type = "normal"
        self.dataset = "night2day"  # cityscapes, night2day, edges2handbags, edges2shoes, facades, maps
        self.num_threads = 0
        self.dataset_mode = "aligned"  # unaligned | aligned | single | colorization
        self.dataroot = f"../data/{self.dataset}"
        # /kaggle/input/{self.dataset}/{self.dataset} ../data/{self.dataset}
        self.phase = "train"
        self.direction = "BtoA"
        self.preprocess = "resize_and_crop"  # [resize_and_crop | crop | scale_width | scale_width_and_crop | none]
        self.batch_size = 1
        self.load_size = 286
        self.crop_size = 256
        self.shuffle = True
        self.no_flip = True
        self.lambda_identity = 0.5
        self.lambda_A = 3
        self.lambda_B = 3
        self.lambda_content = 0.5
        self.lambda_background_mask = 0.1


class TestConfig:
    def __init__(self):
        self.num_threads = 0
        self.batch_size = 1
        self.input_nc = 3
        self.output_nc = 3
        self.shuffle = False
        self.no_flip = True
        self.init_gain = 0.02
        self.pool_size = 0
        self.ndf = 32
        self.gan_mode = "lsgan"  # 'vanilla', 'wgangp', "lsgan"
        self.init_type = "normal"
        self.result_dir = "test_result"
        self.experiment_name = "baseline_maps"
        self.phase = "val"
        self.pretrain_weight_G_A = f"train_result/{self.experiment_name}/net_G_A_latest.pth"
        self.pretrain_weight_G_B = f"train_result/{self.experiment_name}/net_G_B_latest.pth"
        self.pretrain_weight_D_A = f"train_result/{self.experiment_name}/net_D_A_latest.pth"
        self.pretrain_weight_D_B = f"train_result/{self.experiment_name}/net_D_B_latest.pth"
        self.dataset_mode = "aligned"
        self.dataset = "maps"
        self.dataroot = f"../data/{self.dataset}"
        self.crop_size = 256
        self.load_size = 286
        self.direction = "BtoA"
        self.preprocess = "resize_and_crop"
        self.gpu_ids = [0] if torch.cuda.is_available() else [-1]