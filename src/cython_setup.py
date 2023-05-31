from distutils.core import Extension, setup
from Cython.Build import cythonize

# define an extension that will be cythonized and compiled
ext = Extension(name="distance_fast", sources=["distance_fast.pyx"],
                extra_compile_args=['-O3', '-ffast-math', '-march=native'],
                language="c++")
setup(ext_modules=cythonize(ext))
