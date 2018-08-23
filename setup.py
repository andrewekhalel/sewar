from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='sewar',
      version='0.1',
      description='All image quality metrics you need in one package.',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Text Processing :: Linguistic',
      ],
      keywords='image quality performance metric measure ergas q psnr',
      url='http://github.com/storborg/funniest',
      author='Andrew Khalel',
      author_email='andrewekhalel@gmail.com',
      license='MIT',
      packages=['sewar'],
      install_requires=[
          'numpy',
      ],
      zip_safe=False)
