import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("LICENSE", "r", encoding="utf-8") as fh:
    license_file = fh.read()

setuptools.setup(
    name="Haru4me",
    version="0.0.1",
    author="Голов В.А.",
    author_email="golov.v.a@yandex.ru",
    description="Прогнозирование ЗПВ методами нейронных дифференциальных уравнений",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Haru4me/NeuralODE",
    project_urls={
        "Bug Tracker": "https://github.com/Haru4me/NeuralODE",
    },
    license=license_file,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=['torch>=1.3.0', 'scipy>=1.4.0'],
    python_requires=">=3.6",
)
