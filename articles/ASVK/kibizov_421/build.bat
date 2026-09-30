@echo off
rem Сборка статьи: pdflatex дважды (второй проход проставляет номера ссылок
rem на литературу), удаление вспомогательных файлов и упаковка в submit\
rem того, что отправляется в сборник.
rem --enable-installer: MiKTeX сам докачает недостающие пакеты без диалога.
chcp 65001 >nul
cd /d "%~dp0"

for /L %%i in (1,1,2) do (
    pdflatex --enable-installer -interaction=nonstopmode -halt-on-error main_preview.tex >nul
    if errorlevel 1 (
        echo Ошибка сборки, подробности в main_preview.log
        exit /b 1
    )
)

del /q main_preview.aux main_preview.log main_preview.out main_preview.toc build.log 2>nul

if not exist submit mkdir submit
copy /y main_preview.pdf submit\kibizov_421.pdf >nul
copy /y kibizov_421_annotation.txt submit\ >nul
del /q submit\kibizov_421.zip 2>nul
powershell -NoProfile -Command "Compress-Archive -Path kibizov_421.tex, kibizov_421_*.pdf, kibizov_421_*.eps -DestinationPath submit\kibizov_421.zip"

echo Готово: main_preview.pdf и submit\
