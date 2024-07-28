# Generative AI Basics Tutorial

This repository contains a basic tutorial to help you get started with Generative AI. Generative AI focuses on creating models that can generate new content, such as text, images, and more.

## Table of Contents

- [Understanding Generative AI](#understanding-generative-ai)
- [Setting Up Your Environment](#setting-up-your-environment)
- [Building a Simple Text Generator with Transformers](#building-a-simple-text-generator-with-transformers)
- [Creating a Simple Image Generator with GANs](#creating-a-simple-image-generator-with-gans)
- [Experiment and Learn](#experiment-and-learn)
- [Resources](#resources)

## Understanding Generative AI

Generative AI models learn patterns from existing data and use those patterns to generate new, similar content. Some popular types of generative models include:

- **Generative Adversarial Networks (GANs)**: Used for generating images, videos, and other types of data.
- **Variational Autoencoders (VAEs)**: Used for generating new data points in a continuous space.
- **Transformers**: Used for generating text and other sequential data.

## Setting Up Your Environment

Before diving into code, you'll need to set up your environment. Here's a quick guide:

1. **Install Python**: Ensure you have Python installed. You can download it from [python.org](https://www.python.org/downloads/).

2. **Create a Virtual Environment**: This helps manage dependencies.
    ```bash
    python -m venv genai-env
    source genai-env/bin/activate  # On Windows, use `genai-env\Scripts\activate`
    ```

3. **Install Necessary Libraries**: You'll need libraries like TensorFlow, PyTorch, and Transformers.
    ```bash
    pip install tensorflow torch transformers
    ```

## Building a Simple Text Generator with Transformers

We'll use the Hugging Face `transformers` library to build a simple text generator.

1. **Install the `transformers` library**:
    ```bash
    pip install transformers
    ```

2. **Write the Code**:
    ```python
    from transformers import pipeline

    # Load the pre-trained model
    generator = pipeline('text-generation', model='gpt-2')

    # Generate text
    prompt = "Once upon a time,"
    generated_text = generator(prompt, max_length=50, num_return_sequences=1)

    # Print the generated text
    print(generated_text[0]['generated_text'])
    ```

3. **Run the Code**:
    ```bash
    python text_generator.py
    ```

## Creating a Simple Image Generator with GANs

We'll use the PyTorch library to create a basic GAN for generating images.

1. **Install PyTorch**:
    ```bash
    pip install torch torchvision
    ```

2. **Write the Code**:
    ```python
    import torch
    from torch import nn
    from torchvision.utils import save_image
    import numpy as np

    # Define the Generator
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(100, 256),
                nn.ReLU(True),
                nn.Linear(256, 512),
                nn.ReLU(True),
                nn.Linear(512, 1024),
                nn.ReLU(True),
                nn.Linear(1024, 28 * 28),
                nn.Tanh()
            )

        def forward(self, x):
            x = self.model(x)
            x = x.view(x.size(0), 1, 28, 28)
            return x

    # Generate random noise
    noise = torch.randn(64, 100)

    # Initialize the generator
    generator = Generator()

    # Generate images
    generated_images = generator(noise)

    # Save the images
    save_image(generated_images, 'generated_images.png', nrow=8, normalize=True)
    ```

3. **Run the Code**:
    ```bash
    python image_generator.py
    ```

## Experiment and Learn

Generative AI is a vast field with many possibilities. Here are some ways to dive deeper:

- **Experiment with Different Models**: Try different architectures like VAEs or advanced GANs.
- **Use Different Datasets**: Explore different datasets to train your models.
- **Learn from the Community**: Join communities like [Hugging Face](https://huggingface.co/), [Kaggle](https://www.kaggle.com/), and [GitHub](https://github.com/) to learn from others.

## Resources

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [TensorFlow Guides](https://www.tensorflow.org/learn)
- [Generative Adversarial Networks (GANs) in TensorFlow](https://www.tensorflow.org/tutorials/generative/dcgan)

Generative AI is a fascinating and rapidly evolving field. Start with these basics, experiment with different models, and explore the endless possibilities it offers.
