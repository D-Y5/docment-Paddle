@echo off
REM SmartDoc 数据集结构探索工具 (Windows)

echo ========================================
echo SmartDoc 2015 数据集结构探索
echo ========================================
echo.

set DATA_ROOT=C:\Users\dengyu\scikit_learn_data\smartdoc15-ch1

echo 数据集路径: %DATA_ROOT%
echo.

if not exist "%DATA_ROOT%" (
    echo 错误：路径不存在！
    echo 请修改脚本中的 DATA_ROOT 变量为正确的路径
    pause
    exit /b 1
)

echo.
echo [1/4] 显示根目录内容:
echo ----------------------------------------
dir /B "%DATA_ROOT%"
echo.

echo.
echo [2/4] 显示目录树状结构:
echo ----------------------------------------
tree /F "%DATA_ROOT%" > "%TEMP%\smartdoc_tree.txt"
type "%TEMP%\smartdoc_tree.txt" | more
echo.

echo.
echo [3/4] 查找图像文件:
echo ----------------------------------------
echo PNG 文件:
dir /S /B "%DATA_ROOT%\*.png" 2>nul | find /C ".png"
echo.

echo JPG 文件:
dir /S /B "%DATA_ROOT%\*.jpg" 2>nul | find /C ".jpg"
echo.

echo JPEG 文件:
dir /S /B "%DATA_ROOT%\*.jpeg" 2>nul | find /C ".jpeg"
echo.

echo.
echo [4/4] 查找可能的标注文件:
echo ----------------------------------------
echo TXT 文件:
dir /S /B "%DATA_ROOT%\*.txt" 2>nul
echo.

echo CSV 文件:
dir /S /B "%DATA_ROOT%\*.csv" 2>nul
echo.

echo JSON 文件:
dir /S /B "%DATA_ROOT%\*.json" 2>nul
echo.

echo MAT 文件 (MATLAB):
dir /S /B "%DATA_ROOT%\*.mat" 2>nul
echo.

echo ========================================
echo 详细目录树已保存到:
echo %TEMP%\smartdoc_tree.txt
echo ========================================
echo.

pause
