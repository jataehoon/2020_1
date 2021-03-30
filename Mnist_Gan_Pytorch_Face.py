import torchvision.datasets as dset
import torchvision.transforms as transforms

def main():
    dataroot = 'celebA 데이터셋 경로'

    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(64),
                                   transforms.CenterCrop(64),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

import argparse

    parser = argparse.ArgumentParser(description='Face')
    parser.add_argument('--batch_size',type=int,default=128)
    parser.add_argument('--epoch',type=int,default=50)
    parser.add_argument('--learning_rate',type=float,default=0.0002)
    # Hyperparameter Option

    parser.add_argument('--channels',type=int,default=3)  
    parser.add_argument('--noise',type=int,default=100)   
    parser.add_argument('--feature_g',type=int,default=64) 
    parser.add_argument('--feature_d',type=int,default=64) 
    # Convolution Layer Parameter Option (Data Format)

    args = parser.parse_args()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=True, num_workers=0)

import torch.nn as nn

    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(args.noise, args.feature_g * 8, 4, bias=False),
                nn.BatchNorm2d(args.feature_g * 8),
                nn.ReLU(True),
                # state size. (64*8) x 4 x 4
                nn.ConvTranspose2d(args.feature_g * 8, args.feature_g * 4, 4, bias=False),
                nn.BatchNorm2d(args.feature_g * 4),
                nn.ReLU(True),
                # state size. (64*4) x 8 x 8
                nn.ConvTranspose2d(args.feature_g * 4, args.feature_g * 2, 4, bias=False),
                nn.BatchNorm2d(args.feature_g * 2),
                nn.ReLU(True),
                # state size. (64*2) x 16 x 16
                nn.ConvTranspose2d(args.feature_g * 2, args.feature_g, 4, bias=False),
                nn.BatchNorm2d(args.feature_g),
                nn.ReLU(True),
                # state size. (64) x 32 x 32
                nn.ConvTranspose2d(args.feature_g, args.channel, 4, bias=False),
                nn.Tanh()
                # state size. (3) x 64 x 64
            )

        def forward(self, input):
            return self.main(input)

        def __init__(self):
            super(Discriminator, self).__init__()
            self.main = nn.Sequential(
                # input is (channel) x 64 x 64
                nn.Conv2d(args.channel, args.feature_d, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (64) x 32 x 32
                nn.Conv2d(args.feature_d, args.feature_d* 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(args.feature_d* 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (64*2) x 16 x 16
                nn.Conv2d(args.feature_d* 2, args.feature_d* 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(args.feature_d* 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (64*4) x 8 x 8
                nn.Conv2d(args.feature_d* 4, args.feature_d* 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(args.feature_d* 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (64*8) x 4 x 4
                nn.Conv2d(args.feature_d* 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
                # state size. 1 x 1 x 1
            )

        def forward(self, input):
            return self.main(input)

real_label = 1
fake_label = 0

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # 학습용 데이터셋 이미지 학습
        netD.zero_grad()
        real_cpu = data[0].cuda()
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label).cuda()
        output = netD(real_cpu).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # 생성된 이미지 학습
        noise = torch.randn(b_size, args.noise, 1, 1).cuda()
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        # 최종 Loss 값
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

# For further understanding, visit
# https://honeycomb-makers.tistory.com/18?category=714587