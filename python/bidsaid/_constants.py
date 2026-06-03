"""
Module for constants.

Note: The CLI command for converting edat3 files to text files can be found at the
following link: https://support.pstnet.com/hc/en-us/articles/360020316014-DATA-Using-E-DataAid-exe-with-Command-Line-Interpreters-25323
"""

EDATAAID_PATH = r"C:\Program Files (x86)\PST\E-Prime 3.0\Program\E-DataAid.exe"

EDATAAID_CONTROL_FILE = """Inheritance=NULL
InFile={edat_path}
OutFile={dst_path}
ColFlags=0
ColNames=1
Comments=0
BegCommentLine=0
EndCommentLine=0
DataSeparator={delimiter}
VarSeparator={delimiter}
BegDataLine=0
EndDataLine=0
MissingData=nan
Unicode=0
"""
