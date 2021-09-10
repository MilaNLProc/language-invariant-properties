#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

with open('HISTORY.rst') as requirements_file:
    requirements = requirements_file.read()

test_requirements = [ ]

setup(
    author="Federico Bianchi",
    author_email='f.bianchi@unibocconi.it',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Language Invariant Properties",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='language_invariant_properties',
    name='language_invariant_properties',
    packages=find_packages(include=['language_invariant_properties', 'language_invariant_properties.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/vinid/language_invariant_properties',
    version='0.1.0',
    zip_safe=False,
)
