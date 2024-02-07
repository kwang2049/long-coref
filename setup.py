from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    readme = fh.read()

setup(
    name="long-coref",
    version="0.0.0",
    author="Kexin Wang",
    author_email="kexin.wang.2049@gmail.com",
    description="A benchmark on Document-Aware Passage Retrieval (DAPR).",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/kwang2049/long-coref",
    project_urls={
        "Bug Tracker": "https://github.com/kwang2049/long-coref/issues",
    },
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[],
)
