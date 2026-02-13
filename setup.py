from setuptools import setup, find_packages

setup(
    name="psolr",
    version="0.1.0",
    author="Koustav Dutta",
    author_email="koustavdutta.dgp@gmail.com",
    description="Metaheuristic-Optimized Logistic Regression using Particle Swarm Optimization",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/JEET-123/pso-logistic-regression",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "scikit-learn>=1.2"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research/Management/Corporate",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Machine Learning",
        "Topic :: Scientific/Engineering :: Metaheurestic Algorithms",
        "Topic :: Scientific/Engineering :: Neuroevolutionary Algorithm"
    ],
    license="MIT",
)
