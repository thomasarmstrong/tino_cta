from setuptools import setup

setup(
    name='tino_cta',
    version='0.1',
    description='A useful module',
    author='Tino Michael',
    author_email='tino.michael@cea.fr',
    packages=['tino_cta', 'irf_builder'],
    scripts=['scripts/make_point-source_irfs.py'],
    install_requires=['numpy', 'scipy', 'scikit-learn', 'astropy', 'pandas', 'yaml',
                      'matplotlib', 'gammapy']
)
