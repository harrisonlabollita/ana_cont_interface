from setuptools import find_packages, setup

setup(
    name="triqs_ana_cont_interface",
    version="0.1.0",
    description="TRIQS front-end for the ana_cont analytic continuation library",
    author="Harrison LaBollita",
    packages=find_packages(exclude=["test"]),
    python_requires=">=3.9",
    install_requires=["numpy", "scipy"],
)
