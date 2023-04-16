from setuptools import setup


# https://github.com/pypa/setuptools_scm
use_scm = {"write_to": "proteasome/_version.py"}
setup(use_scm_version=use_scm)