# -*- coding: utf-8 -*-

import coremltools
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True, help='Keras model file.')
    parser.add_argument('-l', '--label', type=str, required=True, help='Class label text file.')
    parser.add_argument('-o', '--output', type=str, default='Model.mlmodel', help='Output MLModel file.')
    args = parser.parse_args()

    model = coremltools.converters.keras.convert(args.model,
                                                 input_names='image',
                                                 image_input_names='image',
                                                 output_names=['classLabelProbs', 'classLabel'],
                                                 class_labels=args.label)
    model.save(args.output)


if __name__ == '__main__':
    main()
