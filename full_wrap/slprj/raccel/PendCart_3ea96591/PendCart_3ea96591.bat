@echo off

call "setup_mingw.bat"

cd .

if "%1"=="" ("%MINGW_ROOT%\mingw32-make.exe"  -j 4 -l 6 -Oline -f PendCart_3ea96591.mk all) else ("%MINGW_ROOT%\mingw32-make.exe"  -j 4 -l 6 -Oline -f PendCart_3ea96591.mk %1)
@if errorlevel 1 goto error_exit

exit /B 0

:error_exit
echo The make command returned an error of %errorlevel%
exit /B 1