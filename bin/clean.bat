set current=%~dp0
cd %current%..
python -m pyquickhelper clean_files  -f .
cd %current%
