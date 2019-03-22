from model import MMD_GAN
import mmd
from ops import tf


class SMMD(MMD_GAN):
    def __init__(self, sess, config, **kwargs):
        super(SMMD, self).__init__(sess, config, **kwargs)

    def set_loss(self, G, images):
        kernel = getattr(mmd, '_%s_kernel' % 'rbf')
        kerGI = kernel(G, images)
        print("Enter to Tf variable scope loss")
        with tf.variable_scope('loss'):
            print("!!!!Enter to Tf variable scope loss")
            self.g_loss = mmd.mmd2(kerGI)
            print(self.g_loss)
            self.d_loss = -self.g_loss
            self.optim_name = 'kernel_loss'
        self.add_scaling()
        print('[*] Loss set')

    def apply_scaling(self, scale):
        self.g_loss = self.g_loss*scale
        self.d_loss = -self.g_loss