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
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),

        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)

# 하이퍼 파라미터


device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 64
image_dim = 28*28*1
batch_size = 32
num_epochs = 50


disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
)
dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=lr*0.5)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.MSELoss()
writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
step = 0

Tensor = torch.FloatTensor

start_time = datetime.datetime.now()

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):


        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        valid = Tensor(real.shape[0], 1).fill_(1.0)
        fake1 = Tensor(real.shape[0], 1).fill_(0.0)


        ###
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, valid)
        disc_fake = disc(fake.detach()).view(-1)
        lossD_fake = criterion(disc_fake, fake1)
        lossD = (lossD_fake + lossD_real) / 2
        disc.zero_grad()
        lossD.backward()
        opt_disc.step()

        output = disc(fake).view(-1)
        lossG = criterion(output, valid)
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()


        if batch_idx ==0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}] \ "
                f"Loss D: {lossD:.4f}, Loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images_ls", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist real Images_ls", img_grid_real, global_step=step
                )

                step +=1

end_time= datetime.datetime.now()
learning_time = end_time - start_time
print(learning_time.microseconds)
print(learning_time.seconds)
