set current=%~dp0
cd %current%..
python -m pyquickhelper clean_files -f . --op pep8
cd %current%