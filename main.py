import torch
import numpy as np
import matplotlib.pyplot as plt

from vgg import VGG11_bn
from hessian import Hessian
from loader import get_loader
from sklearn.manifold import TSNE
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

#%% Parameters
# size of the image (after padding)
im_size = 28

# padding of input
pad = 2

# channels in the input to the network
input_ch = 1

# mean and std of all images
mean = 0.1307
std = 0.3081

# number of classes in the dataset
num_class = 10

# dataset is subsetted  to this number of examples per class
examples_per_class = 13

# seed for subsetting dataset
epc_seed = 0

# batch size
batch_size = 64

# use gpu if possible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#%% Dataset
# transform
transform = transforms.Compose([transforms.Pad(pad),
                                transforms.ToTensor(),
                                transforms.Normalize((mean,),(std,))])

# subsampled dataset
dataset = get_loader(full_dataset=MNIST,
                     examples_per_class=examples_per_class,
                     epc_seed=epc_seed,
                     root='.',
                     train=True,
                     transform=transform,
                     download=True)

# loader
loader = DataLoader(dataset=dataset,
                    drop_last=False,
                    batch_size=batch_size)


#%% Network
# initialize network
model = VGG11_bn(im_size=im_size+2*pad,
                 input_ch=input_ch,
                 num_classes=num_class)

# name of trained model
# epc stands for examples per class
checkpoint = 'drive/DeepnetHessian/'                        \
           + 'dataset=MNIST'                                \
           + '-net=VGG11_bn'                                \
           + '-lr=0p027346797255685746'                     \
           + '-examples_per_class='+str(examples_per_class) \
           + '-num_classes='+str(num_class)                 \
           + '-epc_seed='+str(epc_seed)                     \
           + '-train_seed=0'                                \
           + '-epoch=200.pth'

# load trained model
state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
model.load_state_dict(state_dict)
model = model.to(device)


#%% Decomposition of G
# analyze only a subset of classes
Hess = Hessian(crit='CrossEntropyLoss',
               loader=loader,
               device=device,
               model=model,
               num_classes=num_class,
               hessian_type='G')

res = Hess.compute_G_decomp()


#%% Figure 3 in the paper
data = res['dist']
c = [x[0] for x in res['labels']]

# compute t-SNE embedding
tsne_embedded = TSNE(n_components=2,
                     metric='precomputed',
                     perplexity=num_class-1).fit_transform(data)

# plot delta_c
plt.scatter(tsne_embedded[:num_class,0], tsne_embedded[:num_class,1], s=500, c=c[:num_class], alpha=0.5, cmap=plt.get_cmap('rainbow'))

# plot delta_{c,c'}
plt.scatter(tsne_embedded[num_class:,0], tsne_embedded[num_class:,1], s=50, c=c[num_class:], cmap=plt.get_cmap('rainbow'))


#%% Figure 6 in the paper
# approximate spectrum of G using Lanczos
Hess = Hessian(crit='CrossEntropyLoss',
               loader=loader,
               device=device,
               model=model,
               num_classes=num_class,
               hessian_type='G',
               init_poly_deg=32,
               poly_deg=64,
               spectrum_margin=0.05,
               poly_points=1024)

eigval, eigval_density = Hess.LanczosLoop(denormalize=True)

# plot approximation of spectrum of G
plt.semilogy(eigval, eigval_density)

# plot spectrum of SI, SB, SW and ST
plt.scatter(res['G0_eigval'], res['G0_eigval_density'], c='cyan')
plt.scatter(res['G1_eigval'], res['G1_eigval_density'], c='orange')
plt.scatter(res['G2_eigval'], res['G2_eigval_density'], c='red')
plt.scatter(res['G12_eigval'], res['G12_eigval_density'], c='green')


#%% Figure 7 in the paper
# accurate computation of the outliers
Hess = Hessian(crit='CrossEntropyLoss',
               loader=loader,
               device=device,
               model=model,
               num_classes=num_class,
               hessian_type='G',
               init_poly_deg=32,
               poly_deg=64,
               spectrum_margin=0.05,
               poly_points=1024,
               power_method_iters=32)

_, SG0_eigvals, _ = Hess.SubspaceIteration()

# compare the outliers computed using subspace iteration and SB
plt.plot(np.arange(len(SG0_eigvals)), SG0_eigvals, label='Subspace iteration')
plt.plot(np.arange(len(res['G1_eigval'])), res['G1_eigval'], label='SB')
plt.legend()

