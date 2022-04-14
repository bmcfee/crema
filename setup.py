from setuptools import setup, find_packages

import imp

version = imp.load_source('crema.version', 'crema/version.py')

setup(
    name='crema',
    version=version.version,
    description="Convolutional-recurrent estimators for music analysis",
    author='Brian McFee',
    url='http://github.com/bmcfee/crema',
    download_url='http://github.com/bmcfee/crema/releases',
    packages=find_packages(),
    package_data={'': ['models/*/*.pkl',
                       'models/*/*.h5',
                       'models/*/*.json',
                       'models/*/*.txt']},
    long_description="Convolutional-recurrent estimators for music analysis",
    classifiers=[
        "License :: OSI Approved :: ISC License (ISCL)",
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    keywords='audio music learning',
    license='ISC',
    install_requires=['six',
                      'librosa>=0.8',
                      'jams>=0.3',
                      'scikit-learn>=0.18',
                      'keras>=2.6',
                      'tensorflow>=2.0',
                      'mir_eval>=0.5',
                      'pumpp>=0.4',
                      'h5py>=2.7'],
    extras_require={
        'docs': ['numpydoc', 'sphinx'],
        'tests': ['pytest', 'pytest-cov'],
        'training': ['pescador>=2.0.1', 'muda']
    }
)
