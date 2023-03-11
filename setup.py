from setuptools import setup, find_packages

setup(
    name='fastNfair',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/elizabethnewman/fast-n-fair.git',
    license='MIT',
    author='elizabethnewman',
    author_email='elizabeth.newman@emory.edu',
    description='',
    install_requires=['torch', 'hessQuik>=0.0.2', 'matplotlib', 'numpy', 'scikit-learn']
)
