.PHONY: test build upload-pypi clean

test:
	cd csrc && python setup.py build_ext --inplace
	cd ../
	PYTHONPATH=csrc pytest tests/test_fpscan.py -v
	PYTHONPATH=csrc python -m eval.benchmark

build:
	$(PYTHON) -m build --no-isolation

upload-pypi: build
	$(PYTHON) -m twine upload dist/*

clean:
	rm -rf build dist *.egg-info fpscan/_C*.so .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +