from setuptools import setup

PROJECT = 'vedoTk'
package = [f'{PROJECT}']
package_dir = {f'{PROJECT}': 'src'}

# Extract README.md content
with open('README.md') as f:
    long_description = f.read()

# Installation
setup(name=PROJECT,
      version='23.2.23',
      description='A Python toolbox based on vedo for 3D objects visualization and manipulation.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='R. Enjalbert',
      author_email='robin.enjalbert@inria.fr',
      url='https://github.com/RobinEnjalbert/vedoTk',
      packages=package,
      package_dir=package_dir,
      install_requires=['numpy', 'vedo'])
