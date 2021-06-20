import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import datetime



class Discriminator(nn.Module):
    def __init__(self, channel_img, feature_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(channel_img, feature_d, kernel_size= 4, stride= 2, padding= 1),
            nn.ReLU(),
            nn.Conv2d(feature_d, feature_d * 2,4,2,1),
            nn.ReLU(),
            nn.Conv2d(feature_d * 4, 1, kernel_size= 4, stride= 1, padding= 0),
            nn.ReLU()
        )
        self.linear= nn.Sequential(
            nn.Linear(4096,1),
            nn.Sigmoid(),

        )



    def forward(self, x):
        x= self.disc(x)
        x= x.view(-1)

        return self.linear(x)


class Generator(nn.Module):
    def __init__(self, z_dim, channel_img, feature_g):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(z_dim, 7*7*256, bias=False)
        self.gen = nn.Sequential(
        self._block(256, feature_g * 8, 4, 1, 1),
        self._block(feature_g*8,feature_g*4, 3, 2, 1),
        nn.ConvTranspose2d(feature_g * 4, channel_img, 4, 2, 1),
        nn.Tanh(),
        )

    def _block(self, in_channel, out_channel, kernel, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
            in_channel,
            out_channel,
            kernel,
            stride,
            padding,
            bias=False
            ),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.linear1(x)
        x = x.view(7,7,256)
        return self.gen(x)




def weights_init(model):
    name = model.__class__.__name__
    if name.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif name.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)



# 하이퍼 파라미터


device = "cuda" if torch.cuda.is_available() else "cpu"

lr = 2e-4
z_dim = 100
img_channel = 1
img_size = 28
feature_d = 64
feature_g = 16
batch_size = 32
num_epochs = 5



disc = Discriminator(img_channel, feature_d).to(device)
gen = Generator(z_dim, img_channel , feature_g).to(device)
weights_init(disc)
weights_init(gen)

fixed_noise = torch.randn((batch_size, z_dim, 1, 1)).to(device)

transforms = transforms.Compose(
    [
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ]
)

dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=lr*0.5, betas=(0.5,0.999))
opt_gen = optim.Adam(gen.parameters(), lr=lr,betas=(0.5,0.999))
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
step = 0


gen.train()
disc.train()


start_time = datetime.datetime.now()

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):


        real = real.to(device)
        batch_size = real.shape[0]



        noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_fake + lossD_real) / 2
        disc.zero_grad()
        lossD.backward()
        opt_disc.step()

        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()


        if batch_idx ==0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}] \ "
                f"Loss D: {lossD:.4f}, Loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(real, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images_dc", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist real Images_dc", img_grid_real, global_step=step
                )

                step +=1

end_time= datetime.datetime.now()
learning_time = end_time - start_time
print(learning_time.microseconds)
print(learning_time.seconds)