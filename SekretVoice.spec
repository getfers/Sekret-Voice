# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

project = Path(SPECPATH)
analysis = Analysis(
    [str(project / 'main.py')],
    pathex=[str(project)],
    datas=[
        (str(project / 'logo.png'), '.'),
        (str(project / 'sekret.ico'), '.'),
    ],
    hiddenimports=['PIL._tkinter_finder'],
)
archive = PYZ(analysis.pure)
executable = EXE(
    archive,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    name='SekretVoice',
    icon=str(project / 'sekret.ico'),
    console=False,
    upx=False,
)
