from setuptools import setup

setup(
    name = "RealtimeAlignment",
    version = "0.0.1.dev",
    author = "Haider Abidi, Yi Huang, Akshay Malige, Yihui (Ray) Ren",
    author_email = "sabidi@bnl.gov, yhuang2@bnl.gov, amalige@bnl.gov, yren@bnl.gov",
    description = ("Real-Time Detector Alignment"),
    license = "MIT",
    # keywords = "BSD 3-Clause 'New' or 'Revised' License",
    # url = "https://github.com/pphuangyi/sparse_poi/tree/main",
    packages=['rtal'],
    long_description="Real-time detector alignment -- ROM data generation and algorithm design",
    install_requires=[
        "numpy",
        "torch",
        "tqdm",
        "PyYAML",
    ],
)
