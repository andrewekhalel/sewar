from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='sewar',
      version='0.1',
      description='All image quality metrics you need in one package.',
      long_description=readme(),
      classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Multimedia :: Graphics',
      ],
      keywords='image quality performance metric measure ergas q psnr',
      url='https://github.com/andrewekhalel/sewar',
      author='Andrew Khalel',
      author_email='andrewekhalel@gmail.com',
      license='MIT',
      packages=['sewar'],
      install_requires=[
          'numpy', 'scipy'
      ],
      zip_safe=False)
