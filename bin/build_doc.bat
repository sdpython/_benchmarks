set current=%~dp0
cd %current%..
python -c "from sphinx.cmd.build import build_main;build_main(['-j2','-v','-T','-b','html','-d','dist/doctrees','_doc','dist/html'])"
cd %current%