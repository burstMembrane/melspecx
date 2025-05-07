dev:
	maturin develop
	pyo3-stubgen melspecx $(shell python -c 'import melspecx; from pathlib import Path; print(Path(melspecx.__file__).parent)')
build:
	maturin build
	pyo3-stubgen melspecx $(shell python -c 'import melspecx; from pathlib import Path; print(Path(melspecx.__file__).parent)')
	
test:
	cargo test




