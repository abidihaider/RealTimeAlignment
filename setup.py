from setuptools import setup, find_packages

setup(
    name = "RealtimeAlignment",
    version = "0.0.1.dev",
    author = "Haider Abidi, Yi Huang, Akshay Malige, Yihui (Ray) Ren",
    author_email = "sabidi@bnl.gov, yhuang2@bnl.gov, amalige@bnl.gov, yren@bnl.gov",
    description = ("Real-Time Detector Alignment"),
    license = "MIT",
    # keywords = "BSD 3-Clause 'New' or 'Revised' License",
    # url = "https://github.com/pphuangyi/sparse_poi/tree/main",
    # find_packages picks up rtal.data, rtal.models, rtal.geometry and
    # rtal.datasets; listing 'rtal' alone left every subpackage uninstalled.
    packages=find_packages(include=['rtal', 'rtal.*']),
    long_description="Real-time detector alignment -- ROM data generation and algorithm design",
    install_requires=[
        "numpy",
        "torch",
        "tqdm",
        "PyYAML",
        "pandas",       # training logs
        "matplotlib",   # diagnostics plots
    ],
    extras_require={
        # only needed by onnx_no-residual/ and the QAT training variant
        "onnx": ["onnx", "onnxruntime"],
        "qat":  ["torchao"],
    },
)
