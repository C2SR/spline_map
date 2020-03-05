from setuptools import setup, find_namespace_packages

setup(name='spline_map',
      version='0.1',
      description='Occupancy mapping using splines',
      url='https://github.com/romulortr/spline_map',
      author='Romulo T. Rodrigues',
      licence='GPLv3',
      packages=find_namespace_packages(include=['spline_map.*']),#['spline_map/occupancy'],
      zip_safe=False)
