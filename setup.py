from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

"""
Use this setup.py script to compile and install the cython/C++ extensions for solving global
Structure from Motion translations problems. These allow using C++ functions (linking with
ceres-solver) from python.

Usage:
	python setup.py build_ext --inplace
"""

def get_mfas_extension(config):

	sourcefiles        = ['sfminit/src/_wrapper_mfas.pyx', 'sfminit/src/mfas.cc']
	include_dirs       = ['sfminit/src']
	library_dirs       = []
	libraries          = []
	extra_compile_args = ['-O2']
	define_macros      = []
	header_files       = ['sfminit/src/mfas.h']

	return Extension("_wrapper_mfas", sourcefiles,
		             include_dirs=include_dirs,
		             library_dirs=library_dirs,
		             libraries=libraries,
		             extra_compile_args=extra_compile_args,
		             define_macros=define_macros,
		             depends=header_files,
		             language='c++')


def get_transsolver_extension(config):

	sourcefiles        = ['sfminit/src/_wrapper_transsolver.pyx', 'sfminit/src/trans_solver.cc']
	include_dirs       = [config['eigen']] +  config['python_includes']
	library_dirs       = []
	suite_sparse_libs  = ['lapack', 'ccolamd', 'spqr', 'cholmod', 'colamd','camd', 'amd', 'suitesparseconfig']
	ceres_libs         = ['ceres', 'glog', 'gflags']
	libraries          = ceres_libs + suite_sparse_libs + ['pthread', 'm'] + config['omp_libs']
	extra_compile_args = ['-O2', '-Wno-unused-function'] + config['omp_flags']
	define_macros      = []
	header_files       = ['trans_solver.h']

	return Extension("_wrapper_transsolver", sourcefiles,
		                    include_dirs=include_dirs,
		                    library_dirs=library_dirs,
		                    libraries=libraries,
		                    extra_compile_args=extra_compile_args,
		                    define_macros=define_macros,
		                    depends=header_files,
		                    language='c++')

#############################
# Platform Dependent Config #
#############################

# Ceres requires Suitesparse and Eigen3
# Cython needs to see numpy headers that match your python/numpy libs for the trans solver
# This section is where you should give hints if cython can't find something.
import platform
if platform.system() == 'Darwin':
	# The standard Mac compiler, clang, doesn't support omp yet.
	# I needed the python include to find numpy/arrayobject.h
	config = { 'eigen'           : '/usr/local/opt/eigen/include/eigen3',
	           'omp_libs'        : [],
	           'omp_flags'       : [],
	           'python_includes' : ['/usr/local/lib/python2.7/site-packages/numpy/core/include'],
	         }
else:
	config = { 'eigen'           : '/usr/include/eigen3',
	           'omp_libs'        : ['gomp'],
	           'omp_flags'       : ['-fopenmp'],
	           'python_includes' : [],
	         }

#############################


extensions = [get_mfas_extension(config),
			  get_transsolver_extension(config)
			 ]

setup(
	description  = 'C++ code to solve problems in global SfM',
	author       = 'Kyle Wilson',
	author_email = 'wilsonkl@cs.cornell.edu',
	ext_package  = 'sfminit',
    ext_modules  = cythonize(extensions)
)
