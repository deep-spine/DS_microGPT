from setuptools import setup, find_packages
import os

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="ds-microGPT",
    version="0.1.0",
    author="DeepSpine",
    author_email="kandarpaexe@gmail.com",
    description="A lightweight, JAX-based GPT model focused on transparency, hackability, and minimalism.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/deep-spine/ds-microGPT.git",
    packages=find_packages(),
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
    ],
    keywords="gpt jax transformer deep-learning language-model microGPT deepspine",
    license="MIT",
    zip_safe=False,
)
