# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse

import clip


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001, help='lr')
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=float, default=1.17)
    parser.add_argument('--train_epoch', type=float, default=20)
    parser.add_argument('--augment_epoch', type=float, default=10)
    args = parser.parse_args()
    print(args)

    clip.available_models()
    name = 'RN50'

    model, preprocess = clip.load(name)
    model.eval()



if __name__ == '__main__':
    main()

