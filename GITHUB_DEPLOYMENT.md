# GitHub 部署指南

本指南帮助你将 SmartDoc 文档矫正系统部署到 GitHub。

## 前置条件

- 已安装 Git
- 拥有 GitHub 账号
- 项目已初始化 Git 仓库（已完成）

---

## 部署步骤

### 第一步：在 GitHub 上创建仓库

1. 登录 GitHub (https://github.com)
2. 点击右上角的 `+` 按钮，选择 `New repository`
3. 填写仓库信息：
   - **Repository name**: `smartdoc-alignment`（或你喜欢的名称）
   - **Description**: `基于深度学习的文档四边界定位与透视矫正系统`
   - **Public/Private**: 根据需要选择
   - **Initialize this repository**: ❌ **不要勾选**（我们已经初始化了）
   - **Add .gitignore**: ❌ 不需要（我们已经有了）
   - **Choose a license**: 可选（推荐 MIT 或 Apache-2.0）
4. 点击 `Create repository` 按钮

### 第二步：添加远程仓库

创建完成后，GitHub 会显示仓库的 URL，选择 **HTTPS** 或 **SSH** 方式。

**使用 HTTPS（推荐）：**
```bash
git remote add origin https://github.com/YOUR_USERNAME/smartdoc-alignment.git
```

**使用 SSH：**
```bash
git remote add origin git@github.com:YOUR_USERNAME/smartdoc-alignment.git
```

> **注意**：将 `YOUR_USERNAME` 替换为你的 GitHub 用户名，`smartdoc-alignment` 替换为你的仓库名称。

### 第三步：验证远程仓库

```bash
git remote -v
```

输出应该类似：
```
origin  https://github.com/YOUR_USERNAME/smartdoc-alignment.git (fetch)
origin  https://github.com/YOUR_USERNAME/smartdoc-alignment.git (push)
```

### 第四步：推送代码到 GitHub

**首次推送（设置上游分支）：**
```bash
git push -u origin main
```

> **提示**：
> - `-u` 参数会设置本地分支与远程分支的跟踪关系
> - 后续推送只需要 `git push` 即可

**如果遇到错误：**

**错误 1：远程分支冲突**
```bash
# 强制推送（谨慎使用，会覆盖远程代码）
git push -u origin main --force
```

**错误 2：认证失败（HTTPS）**
```bash
# 使用个人访问令牌（推荐）
# Settings -> Developer settings -> Personal access tokens -> Tokens (classic)
# 生成 token 并使用：
git push -u origin main
# 输入用户名和 token（不是密码）
```

**错误 3：SSH 密钥未配置**
```bash
# 生成 SSH 密钥
ssh-keygen -t ed25519 -C "your_email@example.com"

# 启动 ssh-agent
eval "$(ssh-agent -s)"

# 添加密钥
ssh-add ~/.ssh/id_ed25519

# 复制公钥到 GitHub
cat ~/.ssh/id_ed25519.pub
# Settings -> SSH and GPG keys -> New SSH key
```

### 第五步：验证部署

1. 访问你的 GitHub 仓库页面
2. 检查所有文件是否正确上传
3. 检查 README.md 是否正常显示

---

## （可选）配置 GitHub Pages 部署 Web 界面

如果你想将 Web 界面部署到 GitHub Pages：

### 方式 1：使用 GitHub Actions 自动部署

1. 创建 `web/.github/workflows/deploy.yml`：
```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [ main ]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '18'
          cache: 'npm'
          cache-dependency-path: web/package-lock.json

      - name: Install pnpm
        uses: pnpm/action-setup@v4
        with:
          version: 8

      - name: Install dependencies
        working-directory: ./web
        run: pnpm install

      - name: Build
        working-directory: ./web
        run: pnpm build

      - name: Setup Pages
        uses: actions/configure-pages@v5

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: './web/.next'

      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

2. 在 GitHub 仓库中启用 GitHub Pages：
   - Settings -> Pages
   - Build and deployment -> Source: 选择 `GitHub Actions`

3. 提交并推送：
```bash
git add web/.github/
git commit -m "chore: 添加 GitHub Pages 部署配置"
git push
```

4. 等待几分钟，访问：`https://YOUR_USERNAME.github.io/smartdoc-alignment/`

### 方式 2：手动部署

```bash
cd web
pnpm build
pnpm run export
# 将 web/out 目录的内容手动上传到 GitHub Pages 的 gh-pages 分支
```

---

## 克隆仓库

在另一台机器上获取代码：

```bash
git clone https://github.com/YOUR_USERNAME/smartdoc-alignment.git
cd smartdoc-alignment
```

---

## 常见操作

### 查看远程仓库
```bash
git remote -v
```

### 修改远程仓库 URL
```bash
git remote set-url origin https://github.com/NEW_USERNAME/NEW_REPO.git
```

### 删除远程仓库
```bash
git remote remove origin
```

### 查看分支
```bash
git branch -a
```

### 创建新分支
```bash
git checkout -b feature/new-feature
git push -u origin feature/new-feature
```

### 合并分支
```bash
git checkout main
git merge feature/new-feature
git push
```

---

## 发布 Release

发布正式版本：

1. GitHub 仓库页面 -> `Releases` -> `Create a new release`
2. 填写信息：
   - **Tag version**: `v1.0.0`
   - **Release title**: `v1.0.0 - 首个正式版本`
   - **Description**: 描述本版本的主要更新内容
3. 点击 `Publish release`

---

## 添加协作者

邀请其他人参与开发：

1. Settings -> Collaborators and teams
2. 点击 `Add people`
3. 输入协作者的 GitHub 用户名
4. 选择权限：Read, Write, 或 Admin
5. 发送邀请

---

## 项目徽章

在 README.md 中添加项目徽章：

```markdown
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![GitHub Stars](https://img.shields.io/github/stars/YOUR_USERNAME/smartdoc-alignment?style=social)
```

---

## 最佳实践

1. **定期备份**：定期推送到 GitHub
2. **分支管理**：使用功能分支开发，避免直接在 main 分支修改
3. **提交信息规范**：使用清晰的提交信息（参考 Conventional Commits）
4. **保护主分支**：设置 main 分支的保护规则，需要 PR 和 Review 才能合并
5. **使用 Issues**：记录 bug 和功能需求
6. **使用 Projects**：管理项目进度和任务

---

## 安全建议

1. **不要提交敏感信息**：
   - API keys
   - 密码
   - 配置文件（.env）
   - 数据集路径

2. **使用 `.gitignore`**：
   - 已配置好，会排除大文件和临时文件

3. **定期更新依赖**：
   ```bash
   pip install --upgrade -r requirements.txt
   cd web && pnpm update
   ```

4. **启用安全扫描**：
   - Settings -> Security -> Code scanning alerts

---

## 常见问题

### Q1: 推送失败，提示 "Updates were rejected"
```bash
# 拉取远程更新
git pull origin main --rebase

# 解决冲突后再推送
git push origin main
```

### Q2: 如何删除 GitHub 仓库？
- Settings -> General -> Danger Zone -> Delete this repository

### Q3: 如何重命名仓库？
- Settings -> General -> Repository name -> Rename

### Q4: 如何转移仓库所有权？
- Settings -> General -> Danger Zone -> Transfer ownership

---

## 参考资料

- [GitHub 官方文档](https://docs.github.com)
- [Git 官方文档](https://git-scm.com/doc)
- [GitHub Actions 文档](https://docs.github.com/en/actions)

---

## 下一步

部署完成后，你可以：

1. **在本地继续开发**：
   ```bash
   git pull origin main
   # 修改代码
   git add .
   git commit -m "feat: 添加新功能"
   git push
   ```

2. **在 GitHub 上协作**：
   - 使用 Issues 讨论问题
   - 使用 Pull Requests 合并代码
   - 使用 Projects 管理项目

3. **持续集成**：
   - 配置 GitHub Actions 自动测试
   - 自动部署到生产环境

祝你使用愉快！🚀
