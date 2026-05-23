"""
Create: 2026.05.23
Author: SG.SUH
Python: 3.8.10
PyTorch: 1.14.0
"""

from setuptools import setup

package_name = "camera_subscriber"

setup(name=package_name,
      version="0.0.0",
      packages=[package_name],
      data_files=[("share/ament_index/resource_index/packages", ["resource/" + package_name]), ("share/" + package_name, ["package.xml"])],
      install_requires=["setuptools"],
      zip_safe=True,
      maintainer="root",
      maintainer_email="seukgyo.suh@gmail.com",
      description="TODO: Package description",
      license="TODO: License declaration",
      tests_require=["pytest"],
      entry_points={"console_scripts": ["cam_sub = camera_subscriber.cam_sub:main"]})