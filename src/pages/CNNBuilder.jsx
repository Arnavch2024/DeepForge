import React from 'react';
import BaseBuilder from '../components/builder/BaseBuilder.jsx';

const cnnPalette = [
  { type: 'dataset', label: 'Dataset' },
  { type: 'inputImage', label: 'Input Image' },
  { type: 'randomFlip', label: 'Random Flip' },
  { type: 'randomRotation', label: 'Random Rotation' },
  { type: 'randomZoom', label: 'Random Zoom' },
  { type: 'randomContrast', label: 'Random Contrast' },
  { type: 'conv2d', label: 'Conv2D' },
  { type: 'maxpool', label: 'MaxPool' },
  { type: 'batchnorm', label: 'BatchNorm' },
  { type: 'dropout', label: 'Dropout' },
  { type: 'flatten', label: 'Flatten' },
  { type: 'dense', label: 'Dense' },
  { type: 'pretrained', label: 'Pre-trained Model' },
  { type: 'output', label: 'Output' },
];

const cnnSchemas = {
  dataset: {
    title: 'Dataset',
    description: 'Defines the dataset to be used for training and evaluation. Specify the dataset name and the split percentages for training, validation, and testing.',
    fields: [
      { key: 'name', label: 'Dataset Name', type: 'select', options: ['CIFAR-10', 'MNIST', 'ImageNet', 'Custom'], default: 'CIFAR-10', help: 'Select a standard dataset or choose "Custom" to specify a path.' },
      { key: 'custom_path', label: 'Custom Dataset Path', type: 'text', default: '', help: 'If "Custom" is selected, provide the path to your dataset.' },
      { key: 'train_split', label: 'Train Split (%)', type: 'number', default: 80, help: 'Percentage of data to use for training.' },
      { key: 'validation_split', label: 'Validation Split (%)', type: 'number', default: 10, help: 'Percentage of data to use for validation.' },
      { key: 'test_split', label: 'Test Split (%)', type: 'number', default: 10, help: 'Percentage of data to use for testing.' },
      { key: 'batch_size', label: 'Batch Size', type: 'number', default: 32, help: 'Number of samples per gradient update.' },
    ]
  },
  inputImage: {
    title: 'Image Input',
    description: 'Start here. This defines the shape of images fed into the network. A common choice is 224×224×3 (width × height × channels) for RGB images. If your data is grayscale, channels is 1. Normalization (e.g., scaling pixel values to 0–1) is typically done in preprocessing.',
    fields: [
      { key: 'width', label: 'Width', type: 'number', default: 224, help: 'Horizontal size in pixels. Typical values: 32–512 depending on your dataset.' },
      { key: 'height', label: 'Height', type: 'number', default: 224, help: 'Vertical size in pixels. Keep aspect ratio consistent with Width.' },
      { key: 'channels', label: 'Channels', type: 'number', default: 3, help: 'Color channels (3 = RGB, 1 = grayscale). Must match your data.' },
    ]
  },
  randomFlip: {
    title: 'Random Flip',
    description: 'Randomly flips images horizontally or vertically during training to augment the dataset. This helps the model become invariant to object orientation.',
    fields: [
      { key: 'mode', label: 'Mode', type: 'select', options: ['horizontal', 'vertical', 'horizontal_and_vertical'], default: 'horizontal', help: 'The direction of flipping.' },
    ]
  },
  randomRotation: {
    title: 'Random Rotation',
    description: 'Randomly rotates images by a certain amount during training. This helps the model become invariant to object rotation.',
    fields: [
      { key: 'factor', label: 'Factor', type: 'number', default: 0.2, help: 'A float represented as fraction of 2*pi. For example, 0.1 results in a random rotation in the range [-20% * 2*pi, 20% * 2*pi].' },
    ]
  },
  randomZoom: {
    title: 'Random Zoom',
    description: 'Randomly zooms images in or out during training. This helps the model learn features at different scales.',
    fields: [
      { key: 'height_factor', label: 'Height Factor', type: 'number', default: 0.2, help: 'A float represented as a fraction of the image height. e.g., 0.2 means zoom in/out by up to 20%.' },
      { key: 'width_factor', label: 'Width Factor', type: 'number', default: 0.2, help: 'A float represented as a fraction of the image width. e.g., 0.2 means zoom in/out by up to 20%.' },
    ]
  },
  randomContrast: {
    title: 'Random Contrast',
    description: 'Randomly adjusts the contrast of images during training. This helps the model become robust to lighting and color variations.',
    fields: [
      { key: 'factor', label: 'Factor', type: 'number', default: 0.2, help: 'A float representing the contrast adjustment range. e.g., 0.2 means adjust contrast by a random factor in [1-0.2, 1+0.2].' },
    ]
  },
  conv2d: {
    title: 'Conv2D Layer',
    description: 'Learns visual features using sliding filters over the image (or previous feature maps). Early layers detect edges and textures; deeper layers capture shapes and objects. More filters increase representational power but also computation and parameters.',
    fields: [
      { key: 'filters', label: 'Filters', type: 'number', default: 32, help: 'How many filters (feature maps) to learn. Common: 16–256. More filters = more capacity.' },
      { key: 'kernel', label: 'Kernel Size', type: 'text', default: '3x3', help: 'Filter window size, e.g., 3x3 or 5x5. Larger kernels look at bigger neighborhoods.' },
      { key: 'stride', label: 'Stride', type: 'text', default: '1x1', help: 'How far the filter moves each step (e.g., 1x1 or 2x2). Larger stride downsamples more aggressively.' },
      { key: 'padding', label: 'Padding', type: 'select', options: ['valid', 'same'], default: 'same', help: '"same" keeps spatial size (with padding). "valid" performs no padding and shrinks outputs.' },
      { key: 'activation', label: 'Activation', type: 'select', options: ['relu', 'sigmoid', 'tanh', 'gelu'], default: 'relu', help: 'Non-linear function after the convolution. ReLU is a strong default.' },
    ]
  },
  maxpool: {
    title: 'Max Pooling',
    description: 'Reduces spatial size by taking the maximum in each window. This summarizes features and makes the network translation-invariant. Often used after Conv2D layers to downsample.',
    fields: [
      { key: 'pool', label: 'Pool Size', type: 'text', default: '2x2', help: 'Window size (e.g., 2x2). 2x2 is the most common choice.' },
      { key: 'stride', label: 'Stride', type: 'text', default: '2x2', help: 'Distance between pooling windows. Often equals Pool Size.' },
    ]
  },
  batchnorm: {
    title: 'Batch Normalization',
    description: 'Stabilizes and speeds up training by normalizing activations. Often placed after Conv/Dense and before the activation. Helps with deeper networks and higher learning rates.',
    fields: [
      { key: 'momentum', label: 'Momentum', type: 'number', default: 0.99, help: 'Controls how quickly running statistics update. 0.9–0.99 are typical.' },
      { key: 'epsilon', label: 'Epsilon', type: 'number', default: 0.001, help: 'Small constant to avoid division by zero. Leave default unless you see instability.' },
    ]
  },
  dropout: {
    title: 'Dropout',
    description: 'Randomly drops (zeros) a fraction of activations during training. This reduces overfitting and improves generalization. Use more dropout for small datasets or very large models.',
    fields: [
      { key: 'rate', label: 'Rate', type: 'number', default: 0.5, help: 'Fraction of units dropped. 0.1–0.5 is common (0.5 is strong regularization).' },
    ]
  },
  flatten: {
    title: 'Flatten',
    description: 'Converts 2D/3D feature maps into a single 1D vector so you can connect to Dense layers for classification or regression.',
    fields: []
  },
  dense: {
    title: 'Dense Layer',
    description: 'A fully connected layer mixes all input features using learned weights and biases. Typically used near the end for classification heads.',
    fields: [
      { key: 'units', label: 'Units', type: 'number', default: 128, help: 'Number of neurons. Larger adds capacity but risks overfitting.' },
      { key: 'activation', label: 'Activation', type: 'select', options: ['relu', 'sigmoid', 'tanh', 'softmax', 'gelu'], default: 'relu', help: 'Non-linear function. For the last layer in classification use softmax/sigmoid.' },
    ]
  },
  output: {
    title: 'Output Layer',
    description: 'Produces the final predictions. For multi-class classification, use the number of classes with softmax. For binary labels, use 1 unit with sigmoid.',
    fields: [
      { key: 'classes', label: 'Classes', type: 'number', default: 10, help: 'Target classes for classification tasks.' },
      { key: 'activation', label: 'Activation', type: 'select', options: ['softmax', 'sigmoid'], default: 'softmax', help: 'Softmax for multi-class; sigmoid for binary/multi-label.' },
    ]
  },
  pretrained: {
    title: 'Pre-trained Model',
    description: 'Use a pre-trained neural network (trained on ImageNet) as a feature extractor. The base model is frozen, and you add custom layers on top. This technique, called transfer learning, works well with limited data and speeds up training.',
    fields: [
      { key: 'model', label: 'Model', type: 'select', options: ['resnet50', 'vgg16', 'mobilenet'], default: 'resnet50', help: 'ResNet50: Good balance of accuracy and speed. VGG16: Simple architecture, slower. MobileNet: Lightweight for mobile devices.' },
      { key: 'trainable', label: 'Trainable', type: 'select', options: ['false', 'true'], default: 'false', help: 'false: Freeze base model (recommended). true: Fine-tune pre-trained weights (requires lower learning rate).' },
    ]
  },
};

// CNN Presets
const cnnPresets = [
  {
    id: 'lenet',
    name: 'LeNet-ish',
    build: () => {
      const nodes = [
        { id: '1', type: 'default', position: { x: 100, y: 100 }, data: { label: 'Input Image', type: 'inputImage', params: { width: 32, height: 32, channels: 1 } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '2', type: 'default', position: { x: 100, y: 200 }, data: { label: 'Conv2D', type: 'conv2d', params: { filters: 6, kernel: '5x5', stride: '1x1', padding: 'valid', activation: 'relu' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '3', type: 'default', position: { x: 100, y: 300 }, data: { label: 'MaxPool', type: 'maxpool', params: { pool: '2x2', stride: '2x2' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '4', type: 'default', position: { x: 100, y: 400 }, data: { label: 'Conv2D', type: 'conv2d', params: { filters: 16, kernel: '5x5', stride: '1x1', padding: 'valid', activation: 'relu' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '5', type: 'default', position: { x: 100, y: 500 }, data: { label: 'MaxPool', type: 'maxpool', params: { pool: '2x2', stride: '2x2' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '6', type: 'default', position: { x: 100, y: 600 }, data: { label: 'Flatten', type: 'flatten', params: {} }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '7', type: 'default', position: { x: 100, y: 700 }, data: { label: 'Dense', type: 'dense', params: { units: 120, activation: 'relu' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '8', type: 'default', position: { x: 100, y: 800 }, data: { label: 'Dense', type: 'dense', params: { units: 84, activation: 'relu' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '9', type: 'default', position: { x: 100, y: 900 }, data: { label: 'Output', type: 'output', params: { classes: 10, activation: 'softmax' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } }
      ];
      const edges = [
        { id: 'e1-2', source: '1', target: '2', animated: true },
        { id: 'e2-3', source: '2', target: '3', animated: true },
        { id: 'e3-4', source: '3', target: '4', animated: true },
        { id: 'e4-5', source: '4', target: '5', animated: true },
        { id: 'e5-6', source: '5', target: '6', animated: true },
        { id: 'e6-7', source: '6', target: '7', animated: true },
        { id: 'e7-8', source: '7', target: '8', animated: true },
        { id: 'e8-9', source: '8', target: '9', animated: true }
      ];
      return { nodes, edges };
    }
  },
  {
    id: 'simple',
    name: 'Simple CNN',
    build: () => {
      const nodes = [
        { id: '1', type: 'default', position: { x: 100, y: 100 }, data: { label: 'Input Image', type: 'inputImage', params: { width: 224, height: 224, channels: 3 } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '2', type: 'default', position: { x: 100, y: 200 }, data: { label: 'Conv2D', type: 'conv2d', params: { filters: 32, kernel: '3x3', stride: '1x1', padding: 'same', activation: 'relu' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '3', type: 'default', position: { x: 100, y: 300 }, data: { label: 'MaxPool', type: 'maxpool', params: { pool: '2x2', stride: '2x2' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '4', type: 'default', position: { x: 100, y: 400 }, data: { label: 'Conv2D', type: 'conv2d', params: { filters: 64, kernel: '3x3', stride: '1x1', padding: 'same', activation: 'relu' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '5', type: 'default', position: { x: 100, y: 500 }, data: { label: 'MaxPool', type: 'maxpool', params: { pool: '2x2', stride: '2x2' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '6', type: 'default', position: { x: 100, y: 600 }, data: { label: 'Flatten', type: 'flatten', params: {} }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '7', type: 'default', position: { x: 100, y: 700 }, data: { label: 'Dense', type: 'dense', params: { units: 128, activation: 'relu' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '8', type: 'default', position: { x: 100, y: 800 }, data: { label: 'Output', type: 'output', params: { classes: 10, activation: 'softmax' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } }
      ];
      const edges = [
        { id: 'e1-2', source: '1', target: '2', animated: true },
        { id: 'e2-3', source: '2', target: '3', animated: true },
        { id: 'e3-4', source: '3', target: '4', animated: true },
        { id: 'e4-5', source: '4', target: '5', animated: true },
        { id: 'e5-6', source: '5', target: '6', animated: true },
        { id: 'e6-7', source: '6', target: '7', animated: true },
        { id: 'e7-8', source: '7', target: '8', animated: true }
      ];
      return { nodes, edges };
    }
  },
  {
    id: 'resnet-transfer',
    name: 'ResNet Transfer Learning',
    build: () => {
      const nodes = [
        { id: '1', type: 'default', position: { x: 100, y: 100 }, data: { label: 'Input Image', type: 'inputImage', params: { width: 224, height: 224, channels: 3 } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '2', type: 'default', position: { x: 100, y: 200 }, data: { label: 'Pre-trained Model', type: 'pretrained', params: { model: 'resnet50', trainable: 'false' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '3', type: 'default', position: { x: 100, y: 300 }, data: { label: 'Dense', type: 'dense', params: { units: 128, activation: 'relu' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '4', type: 'default', position: { x: 100, y: 400 }, data: { label: 'Dropout', type: 'dropout', params: { rate: 0.5 } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '5', type: 'default', position: { x: 100, y: 500 }, data: { label: 'Output', type: 'output', params: { classes: 10, activation: 'softmax' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } }
      ];
      const edges = [
        { id: 'e1-2', source: '1', target: '2', animated: true },
        { id: 'e2-3', source: '2', target: '3', animated: true },
        { id: 'e3-4', source: '3', target: '4', animated: true },
        { id: 'e4-5', source: '4', target: '5', animated: true }
      ];
      return { nodes, edges };
    }
  },
  {
    id: 'vgg-style',
    name: 'VGG-Style Deep CNN',
    build: () => {
      const nodes = [
        { id: '1', type: 'default', position: { x: 100, y: 100 }, data: { label: 'Input Image', type: 'inputImage', params: { width: 224, height: 224, channels: 3 } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '2', type: 'default', position: { x: 100, y: 200 }, data: { label: 'Conv2D', type: 'conv2d', params: { filters: 64, kernel: '3x3', stride: '1x1', padding: 'same', activation: 'relu' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '3', type: 'default', position: { x: 100, y: 300 }, data: { label: 'Conv2D', type: 'conv2d', params: { filters: 64, kernel: '3x3', stride: '1x1', padding: 'same', activation: 'relu' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '4', type: 'default', position: { x: 100, y: 400 }, data: { label: 'MaxPool', type: 'maxpool', params: { pool: '2x2', stride: '2x2' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '5', type: 'default', position: { x: 100, y: 500 }, data: { label: 'Conv2D', type: 'conv2d', params: { filters: 128, kernel: '3x3', stride: '1x1', padding: 'same', activation: 'relu' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '6', type: 'default', position: { x: 100, y: 600 }, data: { label: 'Conv2D', type: 'conv2d', params: { filters: 128, kernel: '3x3', stride: '1x1', padding: 'same', activation: 'relu' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '7', type: 'default', position: { x: 100, y: 700 }, data: { label: 'MaxPool', type: 'maxpool', params: { pool: '2x2', stride: '2x2' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '8', type: 'default', position: { x: 100, y: 800 }, data: { label: 'Conv2D', type: 'conv2d', params: { filters: 256, kernel: '3x3', stride: '1x1', padding: 'same', activation: 'relu' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '9', type: 'default', position: { x: 100, y: 900 }, data: { label: 'Conv2D', type: 'conv2d', params: { filters: 256, kernel: '3x3', stride: '1x1', padding: 'same', activation: 'relu' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '10', type: 'default', position: { x: 100, y: 1000 }, data: { label: 'MaxPool', type: 'maxpool', params: { pool: '2x2', stride: '2x2' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '11', type: 'default', position: { x: 100, y: 1100 }, data: { label: 'Flatten', type: 'flatten', params: {} }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '12', type: 'default', position: { x: 100, y: 1200 }, data: { label: 'Dense', type: 'dense', params: { units: 512, activation: 'relu' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '13', type: 'default', position: { x: 100, y: 1300 }, data: { label: 'Dropout', type: 'dropout', params: { rate: 0.5 } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '14', type: 'default', position: { x: 100, y: 1400 }, data: { label: 'Dense', type: 'dense', params: { units: 512, activation: 'relu' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '15', type: 'default', position: { x: 100, y: 1500 }, data: { label: 'Dropout', type: 'dropout', params: { rate: 0.5 } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '16', type: 'default', position: { x: 100, y: 1600 }, data: { label: 'Output', type: 'output', params: { classes: 1000, activation: 'softmax' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } }
      ];
      const edges = [
        { id: 'e1-2', source: '1', target: '2', animated: true },
        { id: 'e2-3', source: '2', target: '3', animated: true },
        { id: 'e3-4', source: '3', target: '4', animated: true },
        { id: 'e4-5', source: '4', target: '5', animated: true },
        { id: 'e5-6', source: '5', target: '6', animated: true },
        { id: 'e6-7', source: '6', target: '7', animated: true },
        { id: 'e7-8', source: '7', target: '8', animated: true },
        { id: 'e8-9', source: '8', target: '9', animated: true },
        { id: 'e9-10', source: '9', target: '10', animated: true },
        { id: 'e10-11', source: '10', target: '11', animated: true },
        { id: 'e11-12', source: '11', target: '12', animated: true },
        { id: 'e12-13', source: '12', target: '13', animated: true },
        { id: 'e13-14', source: '13', target: '14', animated: true },
        { id: 'e14-15', source: '14', target: '15', animated: true },
        { id: 'e15-16', source: '15', target: '16', animated: true }
      ];
      return { nodes, edges };
    }
  },
  {
    id: 'mobilenet-transfer',
    name: 'MobileNet Transfer Learning',
    build: () => {
      const nodes = [
        { id: '1', type: 'default', position: { x: 100, y: 100 }, data: { label: 'Input Image', type: 'inputImage', params: { width: 224, height: 224, channels: 3 } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '2', type: 'default', position: { x: 100, y: 200 }, data: { label: 'Pre-trained Model', type: 'pretrained', params: { model: 'mobilenet', trainable: 'false' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '3', type: 'default', position: { x: 100, y: 300 }, data: { label: 'Dense', type: 'dense', params: { units: 256, activation: 'relu' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '4', type: 'default', position: { x: 100, y: 400 }, data: { label: 'BatchNorm', type: 'batchnorm', params: { momentum: 0.99, epsilon: 0.001 } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '5', type: 'default', position: { x: 100, y: 500 }, data: { label: 'Dropout', type: 'dropout', params: { rate: 0.3 } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '6', type: 'default', position: { x: 100, y: 600 }, data: { label: 'Dense', type: 'dense', params: { units: 128, activation: 'relu' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } },
        { id: '7', type: 'default', position: { x: 100, y: 700 }, data: { label: 'Output', type: 'output', params: { classes: 5, activation: 'softmax' } }, style: { border: '1px solid #1f2937', borderRadius: 10, padding: 10, background: '#0b1220', color: '#e2e8f0' } }
      ];
      const edges = [
        { id: 'e1-2', source: '1', target: '2', animated: true },
        { id: 'e2-3', source: '2', target: '3', animated: true },
        { id: 'e3-4', source: '3', target: '4', animated: true },
        { id: 'e4-5', source: '4', target: '5', animated: true },
        { id: 'e5-6', source: '5', target: '6', animated: true },
        { id: 'e6-7', source: '6', target: '7', animated: true }
      ];
      return { nodes, edges };
    }
  }
];

export default function CNNBuilder() {
  return (
    <BaseBuilder
      title="CNN Builder"
      palette={cnnPalette}
      storageKey="deepforge:builder:cnn:v1"
      schemas={cnnSchemas}
      builderType="cnn"
      presets={cnnPresets}
    />
  );
} 