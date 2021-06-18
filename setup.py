import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="disaster-tweets-Galeos93",
    version="0.0.1",
    author="Galeos93",
    author_email="alejandro.garcia.ihs@gmail.com",
    description="Project to work on Kaggle's disaster tweets challenge",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Galeos93/disaster_tweets",
    project_urls={
        "Bug Tracker": "https://github.com/Galeos93/disaster_tweets/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "disaster_tweets"},
    packages=setuptools.find_packages(where="disaster_tweets"),
    python_requires=">=3.7",
)