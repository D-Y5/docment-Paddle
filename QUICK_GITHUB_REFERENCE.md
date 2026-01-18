# GitHub 快速参考

## 🚀 快速推送

### 方式 1：使用脚本（推荐）

**Windows:**
```bash
push_to_github.bat
```

**Linux/macOS:**
```bash
chmod +x push_to_github.sh
./push_to_github.sh
```

### 方式 2：手动推送

```bash
# 1. 添加远程仓库（仅首次）
git remote add origin https://github.com/YOUR_USERNAME/smartdoc-alignment.git

# 2. 推送代码
git push -u origin main
```

---

## 📝 日常开发

### 提交更改
```bash
git add .
git commit -m "feat: 添加新功能"
git push
```

### 拉取更新
```bash
git pull origin main
```

### 查看状态
```bash
git status
```

### 查看历史
```bash
git log --oneline --graph --all
```

---

## 🔀 分支管理

### 创建新分支
```bash
git checkout -b feature/feature-name
git push -u origin feature/feature-name
```

### 切换分支
```bash
git checkout main
```

### 合并分支
```bash
git checkout main
git merge feature/feature-name
git push
```

### 删除分支
```bash
# 本地
git branch -d feature/feature-name

# 远程
git push origin --delete feature/feature-name
```

---

## 🏷️ 提交信息规范

### 格式
```
<type>(<scope>): <subject>
```

### 类型 (type)
- `feat`: 新功能
- `fix`: 修复 bug
- `docs`: 文档更新
- `style`: 代码格式（不影响功能）
- `refactor`: 重构
- `test`: 测试
- `chore`: 构建/工具配置

### 示例
```bash
git commit -m "feat(inference): 添加批量推理功能"
git commit -m "fix(web): 修复图片上传失败问题"
git commit -m "docs: 更新 README.md"
```

---

## 🐛 常见问题

### 推送失败
```bash
# 拉取远程更新
git pull origin main --rebase

# 解决冲突后推送
git push
```

### 撤销提交
```bash
# 撤销最后一次提交（保留更改）
git reset --soft HEAD~1

# 撤销最后一次提交（删除更改）
git reset --hard HEAD~1

# 撤销远程提交（谨慎使用）
git push origin main --force
```

### 查看远程仓库
```bash
git remote -v
```

### 修改远程 URL
```bash
git remote set-url origin https://github.com/NEW_USER/NEW_REPO.git
```

---

## 🎨 添加 README 徽章

```markdown
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![GitHub Stars](https://img.shields.io/github/stars/YOUR_USERNAME/smartdoc-alignment?style=social)
```

---

## 📊 查看统计

### 代码贡献
```bash
git shortlog -sn
```

### 文件变更
```bash
git diff --stat
```

### 提交统计
```bash
git log --all --pretty=format:"%h %cd %s" --date=short
```

---

## 🔐 SSH 配置

### 生成密钥
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

### 启动 ssh-agent
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

### 查看公钥
```bash
cat ~/.ssh/id_ed25519.pub
```

### 添加到 GitHub
Settings -> SSH and GPG keys -> New SSH key

---

## 📱 GitHub CLI (可选)

### 安装
- Windows: `winget install --id GitHub.cli`
- macOS: `brew install gh`
- Linux: `sudo apt install gh`

### 登录
```bash
gh auth login
```

### 创建仓库
```bash
gh repo create smartdoc-alignment --public --source=.
```

### 推送代码
```bash
gh repo view --web
```

---

## 🎯 快速命令别名

在 `~/.gitconfig` 中添加：

```ini
[alias]
  st = status
  co = checkout
  br = branch
  ci = commit
  lg = log --oneline --graph --all
  unstage = reset HEAD --
  last = log -1 HEAD
  visual = log --pretty=format:'%h %s' --graph
```

使用：
```bash
git st          # 等同于 git status
git lg          # 查看漂亮的日志
git ci -m "xxx" # 快速提交
```

---

## 📚 有用链接

- [GitHub 官方文档](https://docs.github.com)
- [Git 官方文档](https://git-scm.com/doc)
- [GitHub Skills](https://skills.github.com/)
- [GitHub Actions](https://github.com/features/actions)

---

## ⚡ 快速备忘录

```
git clone <url>           # 克隆仓库
git add .                 # 添加所有更改
git commit -m "msg"       # 提交
git push                  # 推送
git pull                  # 拉取
git checkout -b <name>    # 创建分支
git merge <branch>        # 合并分支
git status                # 查看状态
git log                   # 查看历史
```

---

💡 **提示**: 将此文件收藏，随时查阅！
