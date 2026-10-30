from setuptools import setup, find_packages

setup(
    name="twitter-open-domain-generative-chatbot",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow",
        "numpy",
        "pandas",
        "scikit-learn",
        "nltk"
    ]
)
