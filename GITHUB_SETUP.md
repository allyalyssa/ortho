# GitHub Setup Instructions

Your project is now initialized as a Git repository with an initial commit. Here's how to sync it to GitHub:

## Option 1: Create a New GitHub Repository (Recommended)

### Step 1: Create a GitHub Repository
1. Go to https://github.com/new
2. Repository name: `psycholinguistics-project` (or your preferred name)
3. Description: `Psycholinguistics research project with orthographic density analysis and EEG data processing`
4. Make it **Public** or **Private** (your choice)
5. **Do NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

### Step 2: Connect Local Repo to GitHub
Run these commands in your terminal (from the project directory):

```bash
# Add the GitHub repository as a remote
git remote add origin https://github.com/YOUR_USERNAME/psycholinguistics-project.git

# Replace YOUR_USERNAME with your actual GitHub username
# Example: git remote add origin https://github.com/johndoe/psycholinguistics-project.git
```

### Step 3: Push to GitHub
```bash
# Push the main branch to GitHub
git push -u origin main
```

You'll be prompted for your GitHub username and password (or personal access token).

## Option 2: Use GitHub CLI (If Installed)

If you have GitHub CLI (`gh`) installed:

```bash
# Create a new repository and push in one command
gh repo create psycholinguistics-project --public --source=. --remote=origin --push
```

## Option 3: Use GitHub Desktop (GUI)

1. Download and install GitHub Desktop from https://desktop.github.com/
2. Open GitHub Desktop
3. File → Add Local Repository → Select this project folder
4. Click "Publish repository" in the top right
5. Choose name, description, and visibility
6. Click "Publish repository"

## After Initial Push

Once your repository is on GitHub, you can:

### Make changes and push:
```bash
# Check status
git status

# Add changes
git add .

# Commit changes
git commit -m "Your commit message"

# Push to GitHub
git push
```

### Pull changes from GitHub:
```bash
git pull
```

## Important Notes

- The `data/` directory is excluded from Git via `.gitignore` (EEG data files are large)
- Only the scripts and documentation are tracked
- You can download the EEG data again using `python get_verbal_eeg.py` on any machine

## Current Repository Status

Your repository contains:
- `orthographic_density.py` - Orthographic neighborhood density analysis
- `get_verbal_eeg.py` - EEG data download script
- `requirements.txt` - Python dependencies
- `README.md` - Project documentation
- `SUMMARY.md` - EEG data download summary
- `.gitignore` - Excludes large data files

Total: 6 files committed
