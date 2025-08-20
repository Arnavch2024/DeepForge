import React from 'react';
import BaseBuilder from '../components/builder/BaseBuilder.jsx';

const cnnPalette = [
  { type: 'inputImage', label: 'Input Image' },
  { type: 'conv2d', label: 'Conv2D' },
  { type: 'maxpool', label: 'MaxPool' },
  { type: 'batchnorm', label: 'BatchNorm' },
  { type: 'dropout', label: 'Dropout' },
  { type: 'flatten', label: 'Flatten' },
  { type: 'dense', label: 'Dense' },
  { type: 'output', label: 'Output' },
];

const cnnSchemas = {
  inputImage: {
    title: 'Image Input',
    description: 'Start here. This defines the shape of images fed into the network. A common choice is 224×224×3 (width × height × channels) for RGB images. If your data is grayscale, channels is 1. Normalization (e.g., scaling pixel values to 0–1) is typically done in preprocessing.',
    fields: [
      { key: 'width', label: 'Width', type: 'number', default: 224, help: 'Horizontal size in pixels. Typical values: 32–512 depending on your dataset.' },
      { key: 'height', label: 'Height', type: 'number', default: 224, help: 'Vertical size in pixels. Keep aspect ratio consistent with Width.' },
      { key: 'channels', label: 'Channels', type: 'number', default: 3, help: 'Color channels (3 = RGB, 1 = grayscale). Must match your data.' },
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
};

const cnnPresets = [
  {
    id: 'lenet-ish',
    name: 'LeNet-ish',
    build: (id) => {
      const n1 = { id: id(), position: { x: 50, y: 80 }, type: 'default', data: { label: 'Input Image', type: 'inputImage', params: { width: 32, height: 32, channels: 1 } } };
      const n2 = { id: id(), position: { x: 280, y: 50 }, type: 'default', data: { label: 'Conv2D', type: 'conv2d', params: { filters: 6, kernel: '5x5', stride: '1x1', padding: 'same', activation: 'relu' } } };
      const n3 = { id: id(), position: { x: 520, y: 50 }, type: 'default', data: { label: 'MaxPool', type: 'maxpool', params: { pool: '2x2', stride: '2x2' } } };
      const n4 = { id: id(), position: { x: 760, y: 50 }, type: 'default', data: { label: 'Conv2D', type: 'conv2d', params: { filters: 16, kernel: '5x5', stride: '1x1', padding: 'same', activation: 'relu' } } };
      const n5 = { id: id(), position: { x: 1000, y: 50 }, type: 'default', data: { label: 'MaxPool', type: 'maxpool', params: { pool: '2x2', stride: '2x2' } } };
      const n6 = { id: id(), position: { x: 1240, y: 50 }, type: 'default', data: { label: 'Flatten', type: 'flatten', params: {} } };
      const n7 = { id: id(), position: { x: 1480, y: 50 }, type: 'default', data: { label: 'Dense', type: 'dense', params: { units: 120, activation: 'relu' } } };
      const n8 = { id: id(), position: { x: 1720, y: 50 }, type: 'default', data: { label: 'Dense', type: 'dense', params: { units: 84, activation: 'relu' } } };
      const n9 = { id: id(), position: { x: 1960, y: 50 }, type: 'default', data: { label: 'Output', type: 'output', params: { classes: 10, activation: 'softmax' } } };
      const e = (s, t) => ({ id: id(), source: s.id, target: t.id });
      return { nodes: [n1, n2, n3, n4, n5, n6, n7, n8, n9], edges: [e(n1,n2), e(n2,n3), e(n3,n4), e(n4,n5), e(n5,n6), e(n6,n7), e(n7,n8), e(n8,n9)] };
    }
  },
  {
    id: 'simple-cnn',
    name: 'Simple CNN',
    build: (id) => {
      const n1 = { id: id(), position: { x: 60, y: 120 }, type: 'default', data: { label: 'Input Image', type: 'inputImage', params: { width: 224, height: 224, channels: 3 } } };
      const n2 = { id: id(), position: { x: 320, y: 120 }, type: 'default', data: { label: 'Conv2D', type: 'conv2d', params: { filters: 32, kernel: '3x3', stride: '1x1', padding: 'same', activation: 'relu' } } };
      const n3 = { id: id(), position: { x: 560, y: 120 }, type: 'default', data: { label: 'MaxPool', type: 'maxpool', params: { pool: '2x2', stride: '2x2' } } };
      const n4 = { id: id(), position: { x: 800, y: 120 }, type: 'default', data: { label: 'Flatten', type: 'flatten', params: {} } };
      const n5 = { id: id(), position: { x: 1040, y: 120 }, type: 'default', data: { label: 'Dense', type: 'dense', params: { units: 128, activation: 'relu' } } };
      const n6 = { id: id(), position: { x: 1280, y: 120 }, type: 'default', data: { label: 'Output', type: 'output', params: { classes: 10, activation: 'softmax' } } };
      const e = (s, t) => ({ id: id(), source: s.id, target: t.id });
      return { nodes: [n1, n2, n3, n4, n5, n6], edges: [e(n1,n2), e(n2,n3), e(n3,n4), e(n4,n5), e(n5,n6)] };
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