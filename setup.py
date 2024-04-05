import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "dendron",
    version = "0.1.5",
    author = "Richard Kelley",
    package_dir = {"" : "src"},
    packages = setuptools.find_packages(where="src"),
    python_requires=">=3.7"
)
