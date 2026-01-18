@echo off
REM GitHub 推送脚本 (Windows)

echo ========================================
echo GitHub 代码推送工具
echo ========================================
echo.

REM 检查是否有远程仓库
git remote -v >nul 2>&1
if errorlevel 1 (
    echo 检测到未配置远程仓库
    echo.
    set /p github_username="请输入你的 GitHub 用户名: "
    set /p repo_name="请输入仓库名称 (默认: smartdoc-alignment): "

    if "%repo_name%"=="" set repo_name=smartdoc-alignment

    set /p connection_type="选择连接方式 (1. HTTPS / 2. SSH, 默认: 1): "

    if "%connection_type%"=="2" (
        set remote_url=git@github.com:%github_username%/%repo_name%.git
    ) else (
        set remote_url=https://github.com/%github_username%/%repo_name%.git
    )

    echo.
    echo 添加远程仓库: %remote_url%
    git remote add origin %remote_url%
)

REM 显示当前远程仓库
echo.
echo 当前远程仓库配置：
git remote -v

echo.
set /p confirm="确认推送代码? (y/n): "

if "%confirm%"=="y" (
    echo.
    echo 正在推送到 GitHub...
    echo.

    REM 获取当前分支
    for /f "tokens=*" %%i in ('git branch --show-current') do set current_branch=%%i

    REM 推送代码
    git push -u origin %current_branch%

    if errorlevel 1 (
        echo.
        echo ❌ 推送失败，请检查：
        echo 1. GitHub 用户名和仓库名是否正确
        echo 2. 网络连接是否正常
        echo 3. 是否有认证问题（HTTPS 需要输入 token）
        echo.
        echo 常见解决方案：
        echo   - HTTPS: 使用 GitHub Personal Access Token
        echo   - SSH: 配置 SSH 密钥
        pause
        exit /b 1
    ) else (
        echo.
        echo ✅ 推送成功！
        echo.
        echo 下一步：
        echo 1. 访问你的 GitHub 仓库查看代码
        echo 2. 配置 GitHub Pages（可选）
        echo 3. 添加 README 徽章和文档
    )
) else (
    echo 已取消推送
)

echo.
pause
