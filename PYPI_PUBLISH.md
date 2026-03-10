# Publishing ChamberCheck to PyPI

This guide walks you through publishing the `chambercheck` package to PyPI, which will permanently reserve the name and allow users to install via `pip install chambercheck`.

## Prerequisites

1. **Install build tools:**
   ```powershell
   pip install --upgrade build twine
   ```

2. **Create PyPI account:**
   - Register at https://pypi.org/account/register/
   - Verify your email address

3. **Create API token:**
   - Go to https://pypi.org/manage/account/token/
   - Click "Add API token"
   - Give it a name (e.g., "chambercheck-upload")
   - Scope: "Entire account" (or specific to project after first upload)
   - **Copy the token immediately** (starts with `pypi-...`) — you won't see it again

4. **Configure credentials:**
   
   Create `~/.pypirc` (on Windows: `C:\Users\YourName\.pypirc`):
   ```ini
   [pypi]
   username = __token__
   password = pypi-AgEIcHlwaS5vcmcC...your-full-token-here
   
   [testpypi]
   username = __token__
   password = pypi-AgENdGVzdC5weXBp...your-testpypi-token-here
   ```

   **Note:** `.pypirc` is in `.gitignore` to prevent accidental token exposure.

---

## Option A: Test First on TestPyPI (Recommended)

TestPyPI is a separate instance for testing package uploads without affecting the real PyPI.

1. **Create TestPyPI account:** https://test.pypi.org/account/register/
2. **Create TestPyPI token:** https://test.pypi.org/manage/account/token/
3. **Build the package:**
   ```powershell
   # Clean old builds
   Remove-Item -Recurse -Force dist, build, src\*.egg-info -ErrorAction SilentlyContinue
   
   # Build distribution
   python -m build
   ```

4. **Upload to TestPyPI:**
   ```powershell
   python -m twine upload --repository testpypi dist/*
   ```

5. **Test installation:**
   ```powershell
   # Create test environment
   python -m venv test_env
   .\test_env\Scripts\Activate.ps1
   
   # Install from TestPyPI
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ chambercheck
   
   # Test imports
   python -c "import chambercheck; print(chambercheck.__version__)"
   
   # Clean up
   deactivate
   Remove-Item -Recurse -Force test_env
   ```

---

## Option B: Publish Directly to PyPI

⚠️ **Once published, you cannot edit or delete a version** — only upload new versions.

1. **Verify package contents:**
   ```powershell
   # Clean old builds
   Remove-Item -Recurse -Force dist, build, src\*.egg-info -ErrorAction SilentlyContinue
   
   # Build distribution
   python -m build
   
   # Check package
   python -m twine check dist/*
   ```

2. **Upload to PyPI:**
   ```powershell
   python -m twine upload dist/*
   ```

3. **Verify on PyPI:**
   - Visit https://pypi.org/project/chambercheck/
   - Check package metadata, description, links

4. **Test installation:**
   ```powershell
   pip install chambercheck
   python -c "import chambercheck; print(chambercheck.__version__)"
   ```

---

## Post-Publication Steps

1. **Create a GitHub release:**
   ```powershell
   git tag v0.1.0
   git push origin v0.1.0
   ```
   Then create a release on GitHub with release notes.

2. **Update PyPI token scope** (optional but recommended):
   - Go to https://pypi.org/manage/account/token/
   - Delete the old token
   - Create a new token scoped only to the `chambercheck` project
   - Update your `~/.pypirc` with the new token

3. **Add PyPI badge to README:**
   ```markdown
   [![PyPI version](https://badge.fury.io/py/chambercheck.svg)](https://pypi.org/project/chambercheck/)
   [![Downloads](https://pepy.tech/badge/chambercheck)](https://pepy.tech/project/chambercheck)
   ```

---

## Publishing Updates

When releasing a new version:

1. **Update version in `pyproject.toml`:**
   ```toml
   version = "0.1.1"  # or "0.2.0", "1.0.0", etc.
   ```

2. **Commit and tag:**
   ```powershell
   git add pyproject.toml
   git commit -m "Bump version to 0.1.1"
   git tag v0.1.1
   git push origin main v0.1.1
   ```

3. **Build and upload:**
   ```powershell
   Remove-Item -Recurse -Force dist, build, src\*.egg-info -ErrorAction SilentlyContinue
   python -m build
   python -m twine upload dist/*
   ```

---

## Troubleshooting

**Error: "File already exists"**
- You're trying to upload a version that already exists on PyPI
- Solution: Bump the version number in `pyproject.toml`

**Error: "Invalid or non-existent authentication information"**
- Your token is incorrect or expired
- Solution: Generate a new token and update `~/.pypirc`

**Error: "403 Forbidden"**
- You don't have permission to upload to this package name
- Solution: If first upload, check package name isn't already taken

**Package name already taken:**
- Someone else already owns `chambercheck` on PyPI
- Solution: Choose a different name (e.g., `chambercheck-tool`, `echo-chamber-check`)
- Update `name` in `pyproject.toml` and the import shim accordingly

---

## Security Notes

- ✅ `.pypirc` is in `.gitignore`
- ✅ Never commit API tokens to git
- ✅ Use project-scoped tokens after first upload
- ✅ Rotate tokens periodically
- ✅ Store `.env` files locally only (already in `.gitignore`)

---

## Current Package Status

- **Package name:** `chambercheck`
- **Version:** `0.1.0` (ready for initial release)
- **License:** MIT
- **Repository:** https://github.com/Mathieu-Feraud/ChamberCheck
- **Python support:** 3.9 - 3.13

The package is **ready to publish** ✅
