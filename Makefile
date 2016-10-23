all: checkversion
	python processing.py

install:
	pip install -r requirements.txt

checkversion:
	@python --version | grep -q 3.5 || { echo "Requires Python 3.5 or greater!" && exit 1; }
