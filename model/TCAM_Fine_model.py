import torch
from .base_model import BaseModel_Fine
from . import network, base_function_dface, external_function_dface
from . import TCAM_Fine_network
import itertools
from .loss import AdversarialLoss, PerceptualLoss, StyleLoss

class TCAM_Fine(BaseModel_Fine):
    """This class implements the pluralistic image completion, for 256*256 resolution image inpainting"""
    def name(self):
        return "Fine Image Completion"

    @staticmethod
    def modify_options(parser, is_train=True):
        """Add new options and rewrite default values for existing options"""
        parser.add_argument('--output_scale', type=int, default=4, help='# of number of the output scale')
        if is_train:
            parser.add_argument('--train_paths', type=str, default='two', help='training strategies with one path or two paths')
            parser.add_argument('--lambda_per', type=float, default=1, help='weight for image reconstruction loss')
            parser.add_argument('--lambda_l1', type=float, default=1, help='weight for kl divergence loss')
            parser.add_argument('--lambda_g', type=float, default=0.1, help='weight for generation loss')
            parser.add_argument('--lambda_sty', type=float, default=250, help='weight for generation loss')

        return parser

    def __init__(self, opt, coarse_model):
        """Initial the pluralistic model"""
        BaseModel_Fine.__init__(self, opt)

        self.loss_names = ['app_g', 'ad_g', 'img_d', 'per', 'sty']
        self.visual_names = ['img_m', 'img_truth', 'img_out', 'img_g']
        self.value_names = ['u_m', 'sigma_m', 'u_post', 'sigma_post', 'u_prior', 'sigma_prior']
        self.model_names = ['G', 'D']
        self.distribution = []

        self.coarse_model = coarse_model
        self.net_G_coarse = self.coarse_model.net_G
        self.net_E_coarse = self.coarse_model.net_E
        self.net_G_coarse.eval()
        self.net_E_coarse.eval()

        self.net_G = TCAM_Fine_network.define_g(gpu_ids=opt.gpu_ids)
        self.net_D = TCAM_Fine_network.define_d(gpu_ids=opt.gpu_ids)

        self.net_G = self.net_G.cuda(self.gpu_ids[0])
        self.net_D = self.net_D.cuda(self.gpu_ids[0])
        if self.isTrain:
            self.GANloss = AdversarialLoss(type='nsgan')
            self.L1loss = torch.nn.L1Loss()
            self.per = PerceptualLoss()
            self.sty = StyleLoss()
            self.optimizer_G = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, self.net_G.parameters())), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.AdamW(itertools.chain(filter(lambda p: p.requires_grad, self.net_D.parameters())),lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        self.setup(opt)

    def set_input(self, input, epoch=0):
        """Unpack input data from the data loader and perform necessary pre-process steps"""
        self.input = input
        self.image_paths = self.input['img_path']
        self.img = input['img']
        self.mask = 1 - input['mask']

        if len(self.gpu_ids) > 0:
            self.img = self.img.cuda(self.gpu_ids[0])
            self.mask = self.mask.cuda(self.gpu_ids[0])

        self.img_truth = self.img * 2 - 1

        self.img_m = self.mask * self.img_truth
        self.img_c = (1 - self.mask) * self.img_truth

    def test_one(self):
        """Run forward processing to get the inputs"""
        # encoder process
        self.save_results(self.img_truth, data_name='truth')
        distributions, f = self.net_E_coarse(self.img_m, self.img_c)
        p_distribution, q_distribution = self.get_distribution(distributions)

        z, f = self.get_G_inputs(p_distribution, q_distribution, f)
        result = self.net_G_coarse(z, f)
        result_rec, result_img_g = result.chunk(2)
        result_out = result_img_g * (1 - self.mask) + self.img_truth * self.mask  # for 201

        self.img_g = self.net_G(result_out, self.img_m)  # for 201
        self.img_out = self.img_g * (1 - self.mask) + self.img_truth * self.mask
        self.save_results(self.img_out, data_name='out')

    def test_two(self):
        """Run forward processing to get the inputs"""
        self.save_results(self.img_truth, data_name='truth')
        distributions, f = self.net_E_coarse(self.img_m, self.img_c)
        p_distribution, q_distribution = self.get_distribution(distributions)

        z, f = self.get_G_inputs(p_distribution, q_distribution, f)
        result = self.net_G_coarse(z, f)
        result_rec, result_img_g = result.chunk(2)
        result_out = result_rec * (1 - self.mask) + self.img_truth * self.mask # for 201

        self.img_g = self.net_G(result_out,self.img_m) # for 201
        self.img_out = self.img_g * (1 - self.mask) + self.img_truth * self.mask
        self.save_results(self.img_out, data_name='out')

    def test(self):
        """Run forward processing to get the inputs"""
        self.save_results(self.img_truth, data_name='truth')
        distributions, f = self.net_E_coarse(self.img_m, self.img_c)
        p_distribution, q_distribution = self.get_distribution(distributions)

        z, f = self.get_G_inputs(p_distribution, q_distribution, f)
        result = self.net_G_coarse(z, f)
        result_rec, result_img_g = result.chunk(2)
        result_out = result_rec * (1 - self.mask) + self.img_truth * self.mask  # for 201

        self.img_g = self.net_G(result_out,self.mask[:, 0:1, :, :])
        self.img_out = self.img_g * (1 - self.mask) + self.img_truth * self.mask
        self.save_results(self.img_out, data_name='out')

    def get_distribution(self, distributions):
        """Calculate encoder distribution for img_m, img_c"""
        sum_valid = (torch.mean(self.mask.view(self.mask.size(0), -1), dim=1) - 1e-5).view(-1, 1, 1, 1)
        m_sigma = 1 / (1 + ((sum_valid - 0.8) * 8).exp_())
        p_distribution, q_distribution, kl_rec, kl_g = 0, 0, 0, 0
        self.distribution = []
        for distribution in distributions:
            p_mu, p_sigma, q_mu, q_sigma = distribution
            m_distribution = torch.distributions.Normal(torch.zeros_like(p_mu), m_sigma * torch.ones_like(p_sigma))
            p_distribution = torch.distributions.Normal(p_mu, p_sigma)
            p_distribution_fix = torch.distributions.Normal(p_mu.detach(), p_sigma.detach())
            q_distribution = torch.distributions.Normal(q_mu, q_sigma)

            kl_rec += torch.distributions.kl_divergence(m_distribution, p_distribution)

            self.distribution.append([torch.zeros_like(p_mu), m_sigma * torch.ones_like(p_sigma), p_mu, p_sigma, q_mu, q_sigma])

        return p_distribution, q_distribution

    def get_G_inputs(self, p_distribution, q_distribution, f):
        """Process the encoder feature and distributions for generation network"""
        feature32 = torch.cat([f.chunk(2)[0], f.chunk(2)[0]], dim=0)
        z_p = p_distribution.rsample()
        z_q = q_distribution.rsample()
        z = torch.cat([z_p, z_q], dim=0)
        return z, feature32

    def forward(self):
        """Run forward processing to get the inputs"""
        distributions, f = self.net_E_coarse(self.img_m, self.img_c)
        p_distribution, q_distribution = self.get_distribution(distributions)

        z, f = self.get_G_inputs(p_distribution, q_distribution, f)
        result = self.net_G_coarse(z, f)
        result_rec, result_img_g = result.chunk(2)
        result_out = result_rec * (1 - self.mask) + self.img_truth * self.mask

        self.img_g = self.net_G(result_out,self.mask[:, 0:1, :, :])
        self.img_out = self.img_g * (1 - self.mask) + self.img_truth * self.mask

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator"""
        # Real
        D_real, _ = netD(real)
        D_fake, _ = netD(fake.detach())
        D_loss = (self.GANloss(D_real, True, True) + self.GANloss(D_fake, False, True)) / 2

        D_loss.backward()

        return D_loss

    def backward_D(self):
        """Calculate the GAN loss for the discriminators"""
        base_function_dface._unfreeze(self.net_D)
        self.loss_img_d = self.backward_D_basic(self.net_D, self.img_truth, self.img_g)

    def backward_G(self):
        """Calculate training loss for the generator"""
        base_function_dface._freeze(self.net_D)
        # g loss fake
        D_fake, _ = self.net_D(self.img_g)

        self.loss_ad_g = self.GANloss(D_fake, True, False) * 0.1

        # calculate l1 loss ofr multi-scale outputs
        self.loss_app_g = self.L1loss(self.img_truth, self.img_g) * 1.0
        self.loss_per = self.per(self.img_g, self.img_truth) * 1.0
        self.loss_sty = self.sty(self.img_truth * (1 - self.mask), self.img_g * (1 - self.mask)) * 250.0

        totalG_loss = self.loss_app_g + self.loss_per + self.loss_sty + self.loss_ad_g

        totalG_loss.backward()

    def optimize_parameters(self):
        """update network weights"""
        self.forward()
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # optimize the completion network parameters
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
