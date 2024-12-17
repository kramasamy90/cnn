import os
import sys
import yaml

from torchvision import datasets, transforms

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..','utils')))
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'src')))
from simple_cnn import SimpleCnn
from train_utils import Trainer


if __name__ == '__main__':
    # Load configuration.
    with open('../configs/train_config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Load dataset.
    train_dataset = datasets.FashionMNIST(root='../../../data/fashion_mnist',
                                    train=True,
                                    transform=transforms.ToTensor(),
                                    download=True)

    # Load model.
    model = SimpleCnn()
    trainer = Trainer(model, train_dataset, config)
    result = trainer.train()