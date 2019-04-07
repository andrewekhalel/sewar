from setuptools import setup


def readme():
	with open('README.md') as f:
		return f.read()


setup(name='sewar',
	version='0.4.2',
	description='All image quality metrics you need in one package.',
	long_description=readme(),
	long_description_content_type="text/markdown",
	classifiers=[
	'Development Status :: 2 - Pre-Alpha',
	'License :: OSI Approved :: MIT License',
	'Operating System :: OS Independent',
	'Programming Language :: Python',
	'Programming Language :: Python :: 2',
	'Programming Language :: Python :: 2.6',
	'Programming Language :: Python :: 2.7',
	'Programming Language :: Python :: 3',
	'Programming Language :: Python :: 3.1',
	'Programming Language :: Python :: 3.2',
	'Programming Language :: Python :: 3.3',
	'Programming Language :: Python :: 3.4',
	'Programming Language :: Python :: 3.5',
	'Programming Language :: Python :: 3.6',
	'Topic :: Multimedia :: Graphics'
	],
	keywords='image quality performance metric measure ergas q psnr pansharpening',
	url='https://github.com/andrewekhalel/sewar',
	author='Andrew Khalel',
	author_email='andrewekhalel@gmail.com',
	license='MIT',
	packages=['sewar'],
	test_suite='nose.collector',
	tests_require=['nose','Pillow'],
	install_requires=[
	'numpy', 'scipy' , 'Pillow'
	],
	entry_points="""
    [console_scripts]
    sewar = sewar.command_line:main
    """,
	zip_safe=False)
