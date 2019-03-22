import  os
class Config(object):
    def __init__(self):
        # Optimizer
        self.max_iteration = 15000
        self.beta1 = 0.5
        self.beta2 = 0.9
        self.learning_rate = 0.0001
        self.learning_rate_D = 1
        self.dsteps = 5
        self.gsteps = 1
        self.start_dsteps = 10
        self.clip_grad = True
        self.batch_norm = False

        self.init = 0.02 #init number
        self.batch_size = 64
        self.real_batch_size = 64
        self.output_size = 128 #size of output image

        self.c_dim = 3 #image dimensions
        self.z_dim = 128#latent_noise
        self.df_dim = 64 #disriminator no of channels at first conv layer
        self.dof_dim = 1 #no of disrminator output features
        self.gf_dim = 64 #no of generator channels

        # Directories
        self.checkpoint_dir = "checkpoint"
        self.sample_dir = "sample"
        self.log_dir = "log"
        self.data_dir = "data"

        # training options
        self.is_train = True

        self.log = True
        self.suffix = ""
        self.gpu_mem = .9
        self.no_of_samples = 100000
        self.save_layer_outputs = 0
        self.ckpt_name = ""


        # Decay rates
        self.decay_rate = .8 #decay rate
        self.gp_decay_rate = .8 #decay rate of gradient penalty
        self.sc_decay_rate = 1. #decay of the scaling factor
        self.restart_lr = False #use lr sheduler on 3-sample test
        self.restart_sc = False #ensures the discriminator is injective by adding the input to the feauture
        self.MMD_lr_sheduler = True #Whether to use lr scheduler based on 3-sample test
        self.MMD_sdlr_past_sample = 10#lr scheduler: number of past iterations to keep
        self.MMD_sdlr_num_test = 3 #lr scheduler: number of failures to decrease KID score
        self.MMD_sdlr_freq = 2000 #lr scheduler: frequency of scoring the model

        # discriminator penalties
        self.gradient_penalty= 0.0 #Use gradient penalty if > 0
        self.L2_discriminator_penalty = 0.0 #Use L2 penalty on discriminator features if > 0

        # scaled MMD
        self.with_scaling = True
        self.scaling_coeff = 10.#coeff of scaling
        self.scaling_variant = 'grad'#Add N(0, 10^2) noise to images in scaling

        # multi-gpu training
        self.num_gpus =  len(os.environ.get('CUDA_VISIBLE_DEVICES', "").split(','))