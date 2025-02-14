from setuptools import setup, find_packages

setup(
    name='neuroquanta',
    version='0.1.0',
    description='NeuroQuantaNetwork - Uma rede neural inovadora com função de ativação PulseWave e módulo CosmicResonanceModulator',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='José Icaro',
    author_email='icarojose533@gmail.com',
    packages=find_packages(),
    install_requires=[],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.7',
    project_urls={
        'Source': 'https://github.com/Icarogamer2441/neuroquanta',
        'Bug Reports': 'https://github.com/Icarogamer2441/neuroquanta/issues',
    },
)
