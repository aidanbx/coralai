from setuptools import setup, find_packages

setup(
    name='coralai',  # This is the name of the package
    packages=find_packages(),  # This will automatically find packages in the current directory
    version='0.0.1',
    description='...',
    author='Aidan Arnaud Barbieux',
    author_email='aidanbx@gmail.com',
    url='git@github.com:abarbieu/coralai.git',  # Use the URL to the github repo
    keywords=['pytorch', 'evolutionary-algorithms', 'taichi', 'emergent-behavior', 'artificial-life', 'neural-cellular-automata'],  # Arbitrary keywords
    classifiers=[],
    install_requires=[
        'pytorch-neat @ git+https://github.com/aidanbx/PyTorch-NEAT#egg=pytorch-neat',
        'click==8.1.3',
        'dask==2023.9.3',
        'gym==0.26.2',
        'neat_python==0.92',
        'noise==1.2.2',
        'numpy==1.23.2',
        'pytest==7.4.3',
        'scipy==1.11.3',
        'taichi==1.6.0',
        'taichi_nightly==1.7.0.post20231020',
        'torch==2.2.0.dev20230926',
    ],
)