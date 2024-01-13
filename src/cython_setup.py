from setuptools import setup
from Cython.Build import cythonize

# define an extension that will be cythonized and compiled
ext = setup(name="distance_fast",
            extra_compile_args=['-O3', '-ffast-math', '-march=native'],
            language="c++", ext_modules=cythonize("distance_fast.pyx"))
