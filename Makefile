install:
	@pip install -e .
	@echo "🌵 pip install -e . completed!"

clean:
	@rm -f */version.txt
	@rm -f .DS_Store
	@rm -f .coverage
	@rm -rf */.ipynb_checkpoints
	@rm -Rf build
	@rm -Rf */__pycache__
	@rm -Rf */*.pyc
	@echo "🧽 Cleaned up successfully!"

all: install clean

data:
	@python gp_patient_survey/data.py

repeat:
	@python gp_patient_survey/scheduler.py

app:
	@streamlit run gp_patient_survey/streamlit_app.py

git_merge:
	$(MAKE) clean
	$(MAKE) lint
	@python gp_patient_survey/automation/git_merge.py
	@echo "👍 Git Merge (master) successfull!"

git_push:
	$(MAKE) clean
	$(MAKE) lint
	@python gp_patient_survey/automation/git_push.py
	@echo "👍 Git Push (branch) successfull!"

test:
	@pytest -v tests

# Specify package name
lint:
	@black gp_patient_survey/
