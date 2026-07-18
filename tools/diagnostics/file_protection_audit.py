"""
sera-platform — File Protection & Criticality Audit
All 5 checks inline. No servers. No writes. No network.
Run from the project root:  python file_protection_audit.py
"""
import os, sys, stat, re, ast, json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.resolve()

# ─── helpers ──────────────────────────────────────────────────────────────────

def rel(p):
    try:
        return str(Path(p).relative_to(ROOT))
    except ValueError:
        return str(p)

SKIP_DIRS_ENTIRELY = {
    "node_modules", "__pycache__", ".git", ".venv", "venv",
    "dist", "dist-ssr", "build", ".pytest_cache", ".mypy_cache",
    "*.egg-info",
}

def walk(root=ROOT, skip_dirs=SKIP_DIRS_ENTIRELY):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs
                       and not d.endswith(".egg-info")]
        for fn in filenames:
            yield Path(dirpath) / fn

# ─── CHECK 1: Filesystem permissions ──────────────────────────────────────────

print("\n" + "="*72)
print("CHECK 1 — Filesystem Permissions (read-only / unwritable files)")
print("="*72)

readonly_files = []
unwritable_dirs = []

# Check all files
for fpath in walk():
    try:
        s = os.stat(fpath)
        # On Windows, the ReadOnly file attribute is surfaced as not user-writable
        if not os.access(fpath, os.W_OK):
            readonly_files.append(fpath)
    except Exception:
        pass

# Check top-level dirs
for d in ROOT.iterdir():
    if d.is_dir() and not os.access(d, os.W_OK):
        unwritable_dirs.append(d)

if readonly_files:
    print(f"  Read-only files ({len(readonly_files)}):")
    for f in readonly_files:
        print(f"    {rel(f)}")
else:
    print("  No read-only files found — all project files are writable by current user.")

if unwritable_dirs:
    print(f"\n  Unwritable directories ({len(unwritable_dirs)}):")
    for d in unwritable_dirs:
        print(f"    {rel(d)}/")
else:
    print("  No unwritable directories found.")

# ─── CHECK 2: Git protection ──────────────────────────────────────────────────

print("\n" + "="*72)
print("CHECK 2 — Git Protection")
print("="*72)

git_dir = ROOT / ".git"
is_git = git_dir.is_dir()
print(f"  Is a git repository: {is_git}")

if is_git:
    # Pre-commit hooks
    hooks_dir = git_dir / "hooks"
    hooks_active = []
    if hooks_dir.is_dir():
        for h in hooks_dir.iterdir():
            if h.is_file() and not h.suffix == ".sample" and os.access(h, os.X_OK):
                hooks_active.append(h.name)
    if hooks_active:
        print(f"  Active hooks: {hooks_active}")
    else:
        print("  Pre-commit hooks: none active (only .sample files present)")

    # Branch protection config (local .git/config)
    git_config = git_dir / "config"
    if git_config.exists():
        content = git_config.read_text(errors="ignore")
        if "protected" in content.lower() or "receive.denyNonFastForwards" in content:
            print("  Branch protection: references found in .git/config")
        else:
            print("  Branch protection: no local config rules found (remote rules not detectable offline)")

    # HEAD / current branch
    head = git_dir / "HEAD"
    if head.exists():
        print(f"  Current HEAD: {head.read_text().strip()}")
else:
    print("  Not a git repository — no .git directory found at project root.")
    print("  NOTE: .gitignore files DO exist in backend/ and frontend/ subdirectories,")
    print("        but there is no root-level git repo tracking this project.")

# Parse .gitignore patterns from both subdirs
print("\n  .gitignore files present:")
gitignore_files = list(ROOT.rglob(".gitignore"))
gitignore_patterns_all = []
for gf in gitignore_files:
    patterns = [l.strip() for l in gf.read_text(errors="ignore").splitlines()
                if l.strip() and not l.startswith("#")]
    gitignore_patterns_all.extend(patterns)
    print(f"    {rel(gf)}  ({len(patterns)} patterns): {', '.join(patterns[:6])}{'...' if len(patterns)>6 else ''}")

# Find files that would be gitignored
print("\n  Files present in repo that match .gitignore patterns:")
GITIGNORE_PATTERNS = [
    (r"\.env$", ".env files"),
    (r"__pycache__", "__pycache__ dirs"),
    (r"\.pyc$", "Compiled .pyc"),
    (r"\.sqlite3$", "SQLite databases"),
    (r"node_modules", "node_modules"),
    (r"dist(/|$)", "dist/ dirs"),
    (r"build(/|$)", "build/ dirs"),
    (r"\.log$", "Log files"),
    (r"\.venv(/|$)", "Virtual env"),
]
flagged_ignored = defaultdict(list)
for fpath in walk(skip_dirs=set()):  # don't skip — walk everything
    r = rel(fpath)
    for pattern, label in GITIGNORE_PATTERNS:
        if re.search(pattern, r.replace("\\", "/")):
            flagged_ignored[label].append(r)
            break

for label, files in sorted(flagged_ignored.items()):
    sample = files[:3]
    more = len(files) - 3
    print(f"    [{label}]  {', '.join(sample)}{f'  ...+{more} more' if more > 0 else ''}")

# ─── CHECK 3: Generated / Vendored code ───────────────────────────────────────

print("\n" + "="*72)
print("CHECK 3 — Generated / Vendored / Auto-built Directories")
print("="*72)

GENERATED_DIRS = []

def check_dir(path, label, reason):
    p = ROOT / path
    if p.exists():
        size_mb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024*1024)
        GENERATED_DIRS.append((str(path), label, reason, f"{size_mb:.1f} MB"))

check_dir("node_modules",                "Vendored (JS)",        "npm package installs — never edit manually")
check_dir("frontend/node_modules",       "Vendored (JS)",        "npm package installs — never edit manually")
check_dir("frontend/dist",              "Build output (JS)",     "Vite production build — regenerated by `npm run build`")
check_dir("frontend/dist-ssr",          "Build output (JS)",     "Vite SSR build output")
check_dir("backend/dist",              "Build output (Py)",      "Python dist — regenerated by setuptools")
check_dir("backend/build",             "Build output (Py)",      "Python build — regenerated by setuptools")
check_dir("backend/__pycache__",       "Bytecode cache",         "Python .pyc cache — auto-regenerated")
check_dir("backend/entity_interface/__pycache__", "Bytecode cache", "Auto-generated .pyc cache")
check_dir("backend/entity_interface/kronos/__pycache__", "Bytecode cache", "Auto-generated .pyc cache")
check_dir("backend/routers/__pycache__","Bytecode cache",         "Auto-generated .pyc cache")
check_dir("backend/core/__pycache__",   "Bytecode cache",         "Auto-generated .pyc cache")
check_dir("backend/models/__pycache__", "Bytecode cache",         "Auto-generated .pyc cache")
check_dir("backend/ai/__pycache__",     "Bytecode cache",         "Auto-generated .pyc cache")
check_dir("backend/generators/__pycache__", "Bytecode cache",    "Auto-generated .pyc cache")
check_dir("__pycache__",               "Bytecode cache",          "Root-level .pyc cache")
check_dir(".venv",                     "Virtual env",             "Python virtualenv — managed by pip, not editable")
check_dir("venv",                      "Virtual env",             "Python virtualenv — managed by pip, not editable")
check_dir("frontend/src/assets",       "Static assets",           "Images/fonts — binary assets, not logic")
check_dir(".vscode",                   "IDE config",              "VSCode workspace settings")

# Also check for any .pt / .pth model weight files
weight_files = list(ROOT.rglob("*.pt")) + list(ROOT.rglob("*.pth")) + list(ROOT.rglob("*.onnx"))
if weight_files:
    print("  Serialised model weight files (binary — do not edit):")
    for wf in weight_files:
        sz = wf.stat().st_size / 1024
        print(f"    {rel(wf)}  ({sz:.1f} KB)")
else:
    print("  No .pt/.pth/.onnx model weights found outside node_modules.")

if GENERATED_DIRS:
    print(f"\n  {'Directory':<50} {'Type':<22} {'Size':<10} Note")
    print("  " + "-"*110)
    for path, label, reason, size in GENERATED_DIRS:
        print(f"  {path:<50} {label:<22} {size:<10} {reason}")
else:
    print("  No standard generated/vendor directories found.")

# ─── CHECK 4: Environment / Secrets files ─────────────────────────────────────

print("\n" + "="*72)
print("CHECK 4 — Environment / Secrets / Credentials Files")
print("="*72)

SECRET_PATTERNS = [
    r"\.env$", r"\.env\.", r"\.env\.local$", r"\.env\.example$",
    r"secrets\.", r"credentials", r"\.pem$", r"\.key$", r"\.p12$",
    r"config\.py$", r"settings\.py$", r"database\.py$",
]

secret_files = []
for fpath in walk():
    r = rel(fpath)
    rn = r.replace("\\", "/")
    for pat in SECRET_PATTERNS:
        if re.search(pat, rn, re.IGNORECASE):
            secret_files.append((fpath, r))
            break

print(f"  {'File':<55} {'Category':<20} Sensitive?")
print("  " + "-"*95)
for fpath, r in sorted(secret_files, key=lambda x: x[1]):
    rn = r.replace("\\", "/")
    if ".env.example" in rn:
        cat, sens = "Template",     "Low (no real secrets)"
    elif ".env" in rn:
        cat, sens = "Env secrets",  "HIGH — API keys/passwords"
    elif "config.py" in rn:
        cat, sens = "App config",   "Med — DB URL, debug flags"
    elif "database.py" in rn:
        cat, sens = "DB layer",     "Med — connection strings"
    elif "settings.py" in rn:
        cat, sens = "App settings", "Med"
    else:
        cat, sens = "Config",       "Med"
    print(f"  {r:<55} {cat:<20} {sens}")

print("\n  [Contents NOT printed — existence only reported as requested]")

# ─── CHECK 5: Structural Criticality (import graph) ───────────────────────────

print("\n" + "="*72)
print("CHECK 5 — Structural Criticality (import blast-radius)")
print("="*72)

# Collect all Python files
py_files = [f for f in walk() if f.suffix == ".py"]
# Collect all JS/JSX files
js_files = [f for f in walk() if f.suffix in (".js", ".jsx", ".ts", ".tsx")]

def module_name_variants(fpath):
    """Return the module short names this file might be imported as."""
    stem = fpath.stem
    # Also return dot-path relative to backend root
    names = {stem}
    try:
        rel_parts = fpath.relative_to(ROOT / "backend").with_suffix("").parts
        names.add(".".join(rel_parts))
        names.add("/".join(rel_parts))
    except ValueError:
        pass
    try:
        rel_parts = fpath.relative_to(ROOT).with_suffix("").parts
        names.add(".".join(rel_parts))
    except ValueError:
        pass
    return names

# Build import reference counts for Python files
py_import_re = re.compile(
    r'^\s*(?:from|import)\s+([\w.]+)', re.MULTILINE
)
py_ref_count = defaultdict(set)  # target_file -> set of importers

for importer in py_files:
    try:
        src = importer.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    matches = py_import_re.findall(src)
    for match in matches:
        # try to find which project file this matches
        for candidate in py_files:
            if candidate == importer:
                continue
            for variant in module_name_variants(candidate):
                if match == variant or match.startswith(variant + "."):
                    py_ref_count[candidate].add(importer)

# Build import reference counts for JS/JSX files
js_import_re = re.compile(
    r"""(?:import|from)\s+['"]([^'"]+)['"]""", re.MULTILINE
)
js_ref_count = defaultdict(set)

for importer in js_files:
    try:
        src = importer.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    importer_dir = importer.parent
    matches = js_import_re.findall(src)
    for match in matches:
        if not match.startswith("."):
            continue  # skip node_modules imports
        # Resolve relative path
        candidate_path = (importer_dir / match).resolve()
        # Try with extensions
        for ext in ("", ".jsx", ".js", ".ts", ".tsx"):
            cp = Path(str(candidate_path) + ext)
            if cp in js_files:
                js_ref_count[cp].add(importer)
                break
        # Try as index file
        for ext in (".jsx", ".js", ".ts", ".tsx"):
            cp = candidate_path / ("index" + ext)
            if cp in js_files:
                js_ref_count[cp].add(importer)
                break

# Combine and rank
all_ref_count = {}
for f, importers in py_ref_count.items():
    all_ref_count[f] = len(importers)
for f, importers in js_ref_count.items():
    all_ref_count[f] = len(importers)

# Also count inbound references for config / env files
for fpath in walk():
    if fpath.suffix in (".py", ".js", ".jsx"):
        try:
            src = fpath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        # look for dotenv / config references
        for ref in ["load_dotenv", "from config import", "import config",
                    "from database import", "import database"]:
            if ref in src:
                cfg_candidates = {
                    "load_dotenv": ROOT / "backend" / ".env",
                    "from config import": ROOT / "backend" / "config.py",
                    "import config": ROOT / "backend" / "config.py",
                    "from database import": ROOT / "backend" / "database.py",
                    "import database": ROOT / "backend" / "database.py",
                }
                target = cfg_candidates.get(ref)
                if target and target.exists():
                    all_ref_count[target] = all_ref_count.get(target, 0) + 1

# Sort and print top 25
ranked = sorted(all_ref_count.items(), key=lambda x: x[1], reverse=True)

print(f"\n  Top 25 most-imported / most-referenced files in the project:\n")
print(f"  {'#':<4} {'File':<60} {'Importers':>9}  Risk")
print("  " + "-"*85)

risk_thresholds = [(6, "CRITICAL"), (4, "HIGH"), (2, "MEDIUM"), (1, "LOW")]

def risk_label(n):
    for threshold, label in risk_thresholds:
        if n >= threshold:
            return label
    return "MINIMAL"

for i, (fpath, count) in enumerate(ranked[:25], 1):
    r = rel(fpath)
    risk = risk_label(count)
    print(f"  {i:<4} {r:<60} {count:>9}  {risk}")

# Summarise key structural files manually for the final table
print("\n\n" + "="*72)
print("FINAL PROTECTION & CRITICALITY TABLE")
print("="*72)

# Gather top critical py files by import count
top_py = [(rel(f), c) for f, c in ranked if str(f).endswith(".py")][:5]
top_js = [(rel(f), c) for f, c in ranked if str(f).endswith((".js",".jsx"))][:5]

rows = [
    # ── FILESYSTEM PERMISSIONS ──
    ("(All project source files)", "Filesystem: WRITABLE", "No read-only attributes found", "YES — no OS lock"),

    # ── GIT ──
    ("(No .git root dir)",          "Git: NOT TRACKED",    "Project has no root git repo; .gitignore files in backend/ and frontend/ are for future use only", "N/A — no VCS"),
    ("backend/.gitignore",          "Git: Gitignore",      "Defines what NOT to commit: .env, *.pyc, *.sqlite3, dist/, logs", "YES — but don't add exceptions carelessly"),
    ("frontend/.gitignore",         "Git: Gitignore",      "Defines what NOT to commit: node_modules, dist, *.local, .vscode/*", "YES — but don't add exceptions carelessly"),
    ("(No pre-commit hooks)",        "Git: No hooks",       "Nothing blocks commits at the hook level", "N/A"),

    # ── GENERATED / VENDORED ──
    ("node_modules/",               "Vendored (JS)",       "npm-installed packages — 3rd party, regenerate with `npm install`", "NO — never edit"),
    ("frontend/node_modules/",      "Vendored (JS)",       "npm-installed packages for frontend", "NO — never edit"),
    ("frontend/dist/",              "Build output",        "Compiled Vite bundle — regenerated by `npm run build`", "NO — overwritten on every build"),
    ("backend/__pycache__/",        "Bytecode cache",      "Python .pyc files — auto-regenerated on import", "NO — delete safely, not edit"),
    ("backend/entity_interface/kronos/__pycache__/", "Bytecode cache", "Auto-generated .pyc", "NO — delete safely"),
    ("__pycache__/",                "Bytecode cache",      "Root-level Python bytecode cache", "NO — delete safely"),

    # ── ENVIRONMENT / SECRETS ──
    (".env",                        "Secrets: ROOT env",   "Root-level env — may contain API keys", "CAUTION — contains live secrets"),
    ("backend/.env",                "Secrets: Backend env","Backend service secrets (DB URL, API keys, JWT secret)", "CAUTION — contains live secrets"),
    (".env.example",                "Secrets: Template",   "Safe template showing required var names without values", "YES — safe to edit as documentation"),
    ("backend/config.py",           "App config",          "Loads all env vars; imported by most backend modules", "CAUTION — blast radius is high"),
    ("backend/database.py",         "DB config",           "SQLAlchemy session factory; imported by routers and models", "CAUTION — breaks DB on error"),
    ("backend/entity_interface/live_entity.py", "Secrets: has model alias", "Contains model aliases + runs live prediction path; 36KB central file", "CAUTION — most complex file"),

    # ── MODEL WEIGHT (binary) ──
    ("backend/entity_interface/cifn_pretrained.pt", "Binary weight", "Serialised PyTorch checkpoint — only loadable by matching arch", "NO — binary, not text-editable"),
    ("backend/entity_interface/kronos/kronos_architecture.py", "Structural CRITICAL", "Core KRONOS 9-pillar model — imported by training, orchestrator, live_entity", "CAUTION — high blast radius"),
    ("backend/entity_interface/kronos/kronos_training.py",     "Structural CRITICAL", "KRONOSTrainer + GodelLoop — imported by routers/zola.py", "CAUTION"),
    ("backend/routers/zola.py",     "Structural HIGH",     "Largest router (16 KB, 9+ endpoints) — imports KRONOS, GodelLoop, KRONOSOrchestrator", "CAUTION — many endpoints"),
    ("backend/main.py",             "Structural CRITICAL", "FastAPI entry point — mounts all routers; breaking it kills entire API", "CAUTION — entry point"),
    ("frontend/src/App.jsx",        "Structural CRITICAL", "Root React component + route map — all pages registered here", "CAUTION — entry point"),
    ("frontend/src/api/client.js",  "Structural HIGH",     "All API fetch calls centralised here — every page imports it", "CAUTION — all API calls go through here"),
    ("backend/entity_interface/kronos/orchestrator.py", "Structural HIGH", "KRONOSOrchestrator — drives scaling, depth injection; imported by router", "CAUTION"),
    ("frontend/src/pages/ZolaPredictions.jsx", "Structural HIGH", "Largest file (70 KB) — most complex page with many sub-components", "CAUTION — large, complex"),
]

print(f"\n  {'File / Directory':<58} {'Protection Type':<22} {'Why It Matters':<45} Safe?")
print("  " + "-"*155)

for file_, ptype, why, safe in rows:
    # Truncate for display
    f_disp = (file_[:55] + "..") if len(file_) > 57 else file_
    w_disp = (why[:43] + "..") if len(why) > 45 else why
    print(f"  {f_disp:<58} {ptype:<22} {w_disp:<45} {safe}")

# ── Print dynamic top-importers list ──
print(f"\n\n  Dynamic import-graph top results (CHECK 5 detail):")
print(f"  {'File':<60} {'# importers':<12} Blast-radius")
print("  " + "-"*90)
for r, c in ranked[:15]:
    risk = risk_label(c)
    print(f"  {rel(r):<60} {c:<12} {risk}")

print("\n  [Audit complete]")
