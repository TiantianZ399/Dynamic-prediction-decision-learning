# GitHub setup checklist

1. Unzip the repository package.
2. Review `README.md`, `LICENSE`, and `CITATION.cff`.
3. Replace `<your-github-username>` in `CITATION.cff` with the actual GitHub account.
4. Confirm that no proprietary raw Wind/Boke data are included.
5. Initialize and push:

```bash
cd dynamic-portfolio-rl
git init
git add .
git commit -m "Initial public research release"
git branch -M main
# Create an empty GitHub repo first, then:
git remote add origin git@github.com:<your-github-username>/dynamic-portfolio-rl.git
git push -u origin main
```

Optional GitHub CLI command:

```bash
gh repo create dynamic-portfolio-rl --public --source=. --remote=origin --push
```

Before making the repo public, check with Boke Simu whether the internship acknowledgement, Boke Simu name, Boke City/Wind data-source statement, and derived result tables may be disclosed.
