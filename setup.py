from setuptools import setup, find_packages

setup(
    name='eincasm-python',  # This is the name of the package
    packages=find_packages(),  # This will automatically find packages in the current directory
    version='0.0.1',
    description='Emergent Intelligence in Neural Cellular Automata Slime Molds: High performance simulations of cellular collectives in Python, Taichi, and PyTorch',
    author='Aidan Arnaud Barbieux',
    author_email='aidanbx@gmail.com',
    url='git@github.com:abarbieu/eincasm-pytito.git',  # Use the URL to the github repo
    keywords=['pytorch', 'evolutionary-algorithms', 'taichi', 'emergent-behavior', 'artificial-life', 'neural-cellular-automata'],  # Arbitrary keywords
    classifiers=[],
)