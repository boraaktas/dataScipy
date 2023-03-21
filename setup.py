from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.md") as history_file:
    history = history_file.read()

requirements = [requirement for requirement in open('requirements.txt')]

"""
setup_requirements = [
    "pytest-runner",
]


test_requirements = [
    "pytest>=3",
]
"""

setup(
    author="Bora Aktas",
    author_email="baktas19@ku.edu.tr",
    version='0.0.1',
    python_requires=">=3.5",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: KU License",
        "Natural Language :: English",
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    description="Python Library for some data-science tasks",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="data_science_basics",
    name="DataBasix",
    packages=find_packages(include=["data_science_basics", "data_science_basics.*"]),
    url="https://github.com/boraaktas/data-science-basics",
)