import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='PRI-T',
    version='0.0.1',
    author='Guy Wilson',
    author_email='ghwilson@stanford.edu',
    description='Testing installation of Package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/guyhwilson/PRI-T',
    project_urls = {
        "Bug Tracker": "https://github.com/guyhwilson/PRI-T/issues"
    },
    license='MIT',
    packages=['PRIT'],
    install_requires=['numba', 'numpy', 'scipy', 'scikit-learn'],
)