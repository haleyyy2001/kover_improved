from setuptools import setup, find_packages

setup(
    name="kover",
    version="0.1.0",
    packages=find_packages(where="kover/core"),
    package_dir={"": "kover/core"},
    install_requires=[
        "numpy>=1.21.0",
    ],
    extras_require={
        "gpu": [
            "cupy-cuda12x; platform_system=='Linux' and platform_machine=='x86_64'",
            "cupy-cuda11x; platform_system=='Linux' and platform_machine=='x86_64'",
            "cupy-rocm-5-0; platform_system=='Linux' and platform_machine=='x86_64'",
        ],
    },
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "kover=kover.cli:main",
        ],
    },
) 