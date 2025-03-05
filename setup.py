import setuptools

setuptools.setup(
    name="sw1dto2d",
    packages=setuptools.find_packages(),
    author="Kevin Larnier",
    package_data={
        "": ["*.cfg"],
    },
    install_requires=[
        "geopy",
        "matplotlib",
        "numpy",
        "pyproj",
        "shapely",
        "fiona",
        "pandas",
    ],
    include_package_data=True,
    author_email="",
    description=("SW1DTO2D"),
)
