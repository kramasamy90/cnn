import os
import sys
import yaml

from torchvision import datasets, transforms

from simple_cnn import SimpleCnn
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'utils')))
print(sys.path)
from trainer import Trainer


if __name__ == '__main__':
    # Load configuration.
    with open('train_config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Load dataset.
    train_dataset = datasets.FashionMNIST(root='../../data/fashion_mnist',
                                    train=True,
                                    transform=transforms.ToTensor(),
                                    download=True)

    # Load model.
    model = SimpleCnn()
    trainer = Trainer(model, train_dataset, config)
    trainer.train()