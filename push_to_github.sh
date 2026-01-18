#!/bin/bash
# GitHub 推送脚本

echo "========================================"
echo "GitHub 代码推送工具"
echo "========================================"
echo ""

# 检查是否有远程仓库
if [ -z "$(git remote -v)" ]; then
    echo "检测到未配置远程仓库"
    echo ""
    read -p "请输入你的 GitHub 用户名: " github_username
    read -p "请输入仓库名称 (默认: smartdoc-alignment): " repo_name
    repo_name=${repo_name:-smartdoc-alignment}

    read -p "选择连接方式 (1. HTTPS / 2. SSH, 默认: 1): " connection_type
    connection_type=${connection_type:-1}

    if [ "$connection_type" = "2" ]; then
        remote_url="git@github.com:${github_username}/${repo_name}.git"
    else
        remote_url="https://github.com/${github_username}/${repo_name}.git"
    fi

    echo ""
    echo "添加远程仓库: $remote_url"
    git remote add origin $remote_url
fi

# 显示当前远程仓库
echo ""
echo "当前远程仓库配置："
git remote -v

echo ""
read -p "确认推送代码? (y/n): " confirm

if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
    echo ""
    echo "正在推送到 GitHub..."
    echo ""

    # 检查当前分支
    current_branch=$(git branch --show-current)

    # 推送代码
    if git push -u origin $current_branch 2>&1; then
        echo ""
        echo "✅ 推送成功！"
        echo ""
        echo "下一步："
        echo "1. 访问你的 GitHub 仓库查看代码"
        echo "2. 配置 GitHub Pages（可选）"
        echo "3. 添加 README 徽章和文档"
    else
        echo ""
        echo "❌ 推送失败，请检查："
        echo "1. GitHub 用户名和仓库名是否正确"
        echo "2. 网络连接是否正常"
        echo "3. 是否有认证问题（HTTPS 需要输入 token）"
        echo ""
        echo "常见解决方案："
        echo "  - HTTPS: 使用 GitHub Personal Access Token"
        echo "  - SSH: 配置 SSH 密钥"
        exit 1
    fi
else
    echo "已取消推送"
fi
