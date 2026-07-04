# GitHub Upload Guide â€” `CognitiveAgentLab/FineTuning`

This guide walks you through uploading this entire `FineTuning/` project as a
subfolder inside your existing [CognitiveAgentLab](https://github.com/KarthikeyanDhanakotti/CognitiveAgentLab)
repository.

There are three ways to do it â€” pick the one you're comfortable with.

---

## Option A Â· Command line (recommended)

**Prerequisites:** Git installed, GitHub CLI (`gh`) optional but nice.

```powershell
# 1. Clone your existing repo (do this in a working folder, e.g., D:\repos)
cd D:\repos
git clone https://github.com/KarthikeyanDhanakotti/CognitiveAgentLab.git
cd CognitiveAgentLab

# 2. Make sure you're on main
git checkout main
git pull origin main

# 3. Create the FineTuning subfolder inside the repo
New-Item -ItemType Directory -Force -Path .\FineTuning

# 4. Copy this project into it
Copy-Item -Recurse -Force `
    "C:\Users\kartdh\OneDrive - Microsoft\Desktop\Model\FineTuning\FineTuning\*" `
    ".\FineTuning\"

# 5. Stage everything
git add FineTuning/

# 6. Sanity check what's about to be committed
git status
git diff --stat --cached

# 7. Commit
git commit -m "Add Week 9 Fine-Tuning webinar project"

# 8. Push
git push origin main
```

Once pushed, your project will be live at:

**`https://github.com/KarthikeyanDhanakotti/CognitiveAgentLab/tree/main/FineTuning`**

All the Colab badges in every `README.md` will start working automatically.

---

## Option B Â· GitHub web UI (drag-and-drop)

Good if you don't want to install Git.

1. Go to https://github.com/KarthikeyanDhanakotti/CognitiveAgentLab
2. Click **Add file â†’ Create new file**
3. In the file name field, type: `FineTuning/README.md` â€” the `FineTuning/` prefix
   creates the folder for you
4. Commit that placeholder file
5. Navigate into the new `FineTuning/` folder in the UI
6. Click **Add file â†’ Upload files**
7. Drag the entire contents of
   `C:\Users\kartdh\OneDrive - Microsoft\Desktop\Model\FineTuning\FineTuning\`
   into the upload area â€” **including subfolders**. GitHub preserves folder structure.
8. Scroll down, add commit message: *"Add Week 9 Fine-Tuning webinar project"*
9. Click **Commit changes**

> âš ï¸ The web UI has a **100 files / 25 MB per commit** limit. If you hit it, upload the
> four `module_*` folders in separate commits.

---

## Option C Â· GitHub Desktop

1. Open [GitHub Desktop](https://desktop.github.com/)
2. **File â†’ Clone repository** â†’ select `KarthikeyanDhanakotti/CognitiveAgentLab`
3. In Windows Explorer, copy the entire `FineTuning\` folder from
   `C:\Users\kartdh\OneDrive - Microsoft\Desktop\Model\FineTuning\` into the
   cloned repo folder
4. Back in GitHub Desktop, you'll see all the new files staged
5. Bottom-left: summary = *"Add Week 9 Fine-Tuning webinar project"*
6. Click **Commit to main**
7. Click **Push origin**

---

## After the upload â€” verification checklist

- [ ] Open `https://github.com/KarthikeyanDhanakotti/CognitiveAgentLab/tree/main/FineTuning`
      and confirm all four `module_*` folders are visible.
- [ ] Click any **Open in Colab** badge in the top-level README â€” it should open
      the notebook in Google Colab.
- [ ] Confirm `.gitignore` was uploaded (it starts with a dot, some UIs hide it â€”
      look for it in the file list, not in Windows Explorer's default view).
- [ ] Confirm the `.pptx` in `docs/` uploaded correctly (it's binary â€” GitHub will
      show "View raw" instead of previewing).
- [ ] Check file size â€” the whole `FineTuning/` folder should be **under 15 MB**.
      If it's larger, something like a model checkpoint or dataset got in.

---

## Update the top-level `CognitiveAgentLab/README.md`

Add a link to this project so people can find it from the repo root. Append to the
main README:

```markdown
## Projects

- [`FineTuning/`](FineTuning/) â€” Week 9 webinar: fine-tuning a healthcare LLM
  end-to-end (QLoRA â†’ HF Hub â†’ LangSmith eval) on Google Colab.
```

---

## What NOT to upload

The `.gitignore` in this project already blocks the following, but double-check
they didn't sneak in:

- `.env` files with API keys â€” **never commit these**
- Model checkpoints (`*.safetensors`, `*.bin`, `adapter_model.safetensors` from your
  own training run â€” the ones in `results/` are metadata JSONs, those are fine)
- Colab session artifacts (`sample_data/`, `runs/`)
- `__pycache__/`, `.ipynb_checkpoints/`

If you accidentally commit a secret:

```powershell
# Revoke the leaked key immediately, then rewrite history:
# See https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/
```

Better: don't put keys in files. Use `os.environ` in notebooks and let attendees
paste their own keys interactively.

---

## Updating Colab badges

The badges in every README point at:

```
https://colab.research.google.com/github/KarthikeyanDhanakotti/CognitiveAgentLab/blob/main/projects/FineTuning/<path>
```

If your repo default branch is `master` instead of `main`, do a find/replace:

```powershell
# From inside the FineTuning/ folder before upload:
Get-ChildItem -Recurse -Filter *.md |
  ForEach-Object {
    (Get-Content $_.FullName) -replace '/blob/main/', '/blob/master/' |
      Set-Content $_.FullName
  }
```

Also update the folder name in badges if you decide to name the subfolder something
other than `FineTuning` inside the repo.

---

## Optional: pin a release

Once uploaded, you can tag the state used for the webinar:

```powershell
git tag -a v1.0-webinar -m "Week 9 Fine-Tuning webinar â€” final state"
git push origin v1.0-webinar
```

This makes the exact webinar version citable in future updates.
