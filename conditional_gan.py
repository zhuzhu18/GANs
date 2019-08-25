import torch
import torch.nn as nn
from torch.utils import data
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchsize', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=2e-4, help='adam: learning rate')
parser.add_argument("--beta1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--beta2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument('--latent-dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument('--noise-dim', type=int, default=100, help='dimensionality of the noise')
parser.add_argument('--gpu', action='store_true', default=True, help='use cuda')
parser.add_argument('--n_row', type=int, default=10, help='numbers to show in one line')

opt = parser.parse_args()

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        self.emb = nn.Embedding(opt.n_classes, opt.latent_dim)
        self.model = nn.Sequential(
            *self.block(opt.latent_dim+opt.noise_dim, 128, normalize=False),
            *self.block(128, 256),
            *self.block(256, 512),
            *self.block(512, 1024),
            nn.Linear(1024, opt.img_size*opt.img_size),
            nn.Tanh()
        )

    def block(self, in_feat, out_feat, normalize=True):
        layers = []
        layers.append(nn.Linear(in_feat, out_feat))
        if normalize:
            layers.append(nn.BatchNorm1d(out_feat, eps=0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        return layers

    def forward(self, labels, noise):
        x = torch.cat([self.emb(labels), noise], dim=-1)
        x = self.model(x)
        imgs = x.view(labels.size(0), 1, opt.img_size, opt.img_size)

        return imgs

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.emb = nn.Embedding(opt.n_classes, opt.latent_dim)

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim+1*opt.img_size*opt.img_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1)
        )

    def forward(self,  imgs, labels):
        x = torch.cat([imgs.view(imgs.size(0), -1), self.emb(labels)], dim=-1)

        score = self.model(x)
        return score

criterion = torch.nn.MSELoss()
generator = Generator(opt)
discriminator = Discriminator(opt)
cuda_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if opt.gpu:
    generator.to(cuda_device)
    discriminator.to(cuda_device)

transform = transforms.Compose([
    transforms.Resize(opt.img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
dataloader = data.DataLoader(
    datasets.MNIST('/media/zhuzhu/ec114170-f406-444f-bee7-a3dc0a86cfa2/dataset/MNIST',
                    train=True, download=False, transform=transform),
    batch_size=opt.batchsize, shuffle=True
)

optimizer_g = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
FloatTensor = torch.cuda.FloatTensor if opt.gpu else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if opt.gpu else torch.LongTensor

def sample_image(n_row, img_name):
    z = FloatTensor(np.random.normal(0, 1, (n_row**2, opt.noise_dim)))
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = LongTensor(labels)
    gen_imgs = generator(labels, z)
    save_image(gen_imgs, 'images/%d.png'%img_name, nrow=n_row, normalize=True)

for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        real_imgs = imgs.type(FloatTensor)
        labels = labels.type(LongTensor)
        batch_size = labels.size(0)
        valid = FloatTensor(batch_size, 1).fill_(1)
        fake = FloatTensor(batch_size, 1).zero_()

        # train generator
        optimizer_g.zero_grad()
        z = FloatTensor(np.random.normal(0, 1, size=(batch_size, opt.noise_dim)))
        gen_labels = LongTensor(np.random.randint(0, opt.n_classes, batch_size))
        gen_imgs = generator(gen_labels, z)
        g_loss = criterion(discriminator(gen_imgs, gen_labels), valid)
        g_loss.backward()
        optimizer_g.step()

        # train discriminator
        real_score = discriminator(real_imgs, labels)
        fake_score = discriminator(gen_imgs.detach(), gen_labels)

        d_loss = (criterion(real_score, valid) + criterion(fake_score, fake)) / 2
        d_loss.backward()
        optimizer_d.step()

        print("epoch: %d/%d, batch: %d/%d, Generator loss:%f, Discriminator loss:%f"%
              (epoch, opt.n_epochs, i, len(dataloader), g_loss, d_loss))

        if i % opt.sample_interval == 0:
            sample_image(opt.n_row, epoch*len(dataloader)+i+1)
