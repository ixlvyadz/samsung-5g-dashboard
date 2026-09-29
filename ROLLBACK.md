# Original repository backup and rollback

Before releasing Version 4, the original GitHub `main` state was:

- Repository: https://github.com/rrosoidea-commits/samsung-5g-dashboard
- Commit: `ba782eb49ba443f4641ef2d6159815851084a360`
- Backup branch: `backup/pre-version4-20260929`
- Local complete-history bundle: `backups/pre-version4-20260929.bundle` (ignored by Git)
- Bundle SHA-256: `5d6ebf2dcae1c675a950e1655868b905b360f511681045820c51c373e485bc28`

## Revert the release

Use the Version 4 release commit hash from Git history. With any later local work safely committed or stashed:

```bash
git switch main
git pull --ff-only origin main
git revert <Version-4-release-commit>
git push origin main
```

This creates a new commit undoing the release without rewriting Git history. Later changes may require resolving conflicts. If the Streamlit app is configured to deploy this repository's `main` branch, it will pick up the reverted code through its normal deployment process.

## Inspect the original version separately

```bash
git fetch origin
git worktree add ../samsung-original --detach origin/backup/pre-version4-20260929
```

## Recover without GitHub

```bash
git bundle verify backups/pre-version4-20260929.bundle
git clone backups/pre-version4-20260929.bundle ../samsung-recovered
```

The bundle contains committed repository history, assets, and datasets, not the uncommitted Version 4 changes or virtual environment. Keep a separate copy of the bundle if you need protection against loss of this computer.
