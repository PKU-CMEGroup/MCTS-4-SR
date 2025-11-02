from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent
readme_path = here / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Pin dependencies (correcting scikit_learn -> scikit-learn for PyPI)
install_requires = [
    "numba==0.61.2",
    "numpy==2.2.0",
    "pandas==2.3.0",
    "PyYAML==6.0.1",
    "scikit-learn==1.2.2",
    "nlopt==2.9.1",
    "sortedcontainers==2.4.0",
    "sympy==1.12",
    "tqdm==4.65.0",
]

setup(
    name="imcts",
    version="0.1.0",
    description="iMCTS algorithm for symbolic regression",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="PKU-CMEGroup",
    url="https://github.com/PKU-CMEGroup/MCTS-4-SR",
    license="MIT",
    packages=find_packages(exclude=("assets*", "benchmarks*", "scripts*", "tests*")),
    include_package_data=False,
    install_requires=install_requires,
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    project_urls={
        "Source": "https://github.com/PKU-CMEGroup/MCTS-4-SR",
        "Issues": "https://github.com/PKU-CMEGroup/MCTS-4-SR/issues",
    },
)
