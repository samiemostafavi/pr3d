import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()
setup(
  name="pr3d",
  version="0.0.1",
  description="Prediction of delay density using Tensorflow",
  long_description=README,
  long_description_content_type="text/markdown",
  author="Seyed Samie Mostafavi",
  author_email="samiemostafavi@gmail.com",
  license="MIT",
  packages=find_packages(include=['pr3d','pr3d.de'], exclude=['docker','test','utils']),
  zip_safe=False
)