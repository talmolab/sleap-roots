# EXECUTION-READY PLAN: API Documentation Formatting Fixes

## Status: READY FOR IMPLEMENTATION

**Prepared**: 2025-11-19
**Project**: sleap-roots (branch: docs/add-mkdocs)
**Estimated Duration**: 1.5-2 hours

---

## EXECUTIVE SUMMARY

**Two formatting issues identified and analyzed:**

1. **Duplicate Headings** - Manual markdown headings (e.g., `#### function_name`) placed before mkdocstrings blocks that auto-generate headings, resulting in duplicate headings in rendered output

2. **Broken Bullet Lists** - "See Also" sections with bullet lists positioned incorrectly relative to mkdocstrings blocks, causing improper rendering

**Files to Fix**:
- `/Users/elizabethberrigan/repos/sleap-roots/docs/api/core/series.md` (~650 lines)
- `/Users/elizabethberrigan/repos/sleap-roots/docs/api/core/pipelines.md` (~612 lines)

**Expected Changes**: ~15 manual headings removed, bullet list formatting improved

---

## DETAILED ANALYSIS

### Issue 1: Duplicate Headings Pattern

**Incorrect (current in series.md and pipelines.md)**:
```markdown
#### function_name

::: sleap_roots.Series.function_name
    options:
      show_source: true
      heading_level: 4
```

**Problem**: Two headings generated:
1. Manual `####` heading
2. Auto-generated heading from `heading_level: 4`

**Correct (current in trait modules)**:
```markdown
::: sleap_roots.bases.get_bases
    options:
      show_source: true
```

**Solution**: Remove manual heading, let mkdocstrings auto-generate

### Issue 2: Broken Bullet Lists Pattern

**Incorrect (current)**:
```markdown
::: sleap_roots.Series.load
    options:
      show_source: true

**See Also**:
- [link1](#anchor1)
```

**Problem**: Bullet list immediately follows :::block, rendering breaks

**Correct**:
```markdown
::: sleap_roots.Series.load
    options:
      show_source: true

**Example**:
```python
...
```

**See Also**:

- [link1](#anchor1)
```

**Solution**: Add blank line before bullets, ensure positioning is clear

---

## 6-PHASE EXECUTION PLAN

### Phase 1: Comprehensive Audit (15-20 min)

**1.1 Audit series.md**
```bash
grep -n "^#### " docs/api/core/series.md
grep -n "See Also" docs/api/core/series.md
```
Expected: Identify all manual `####` headings and "See Also" sections

**1.2 Audit pipelines.md**
```bash
grep -n "^## \|^### " docs/api/core/pipelines.md
grep -n "See Also" docs/api/core/pipelines.md
```
Expected: Identify pipeline class headings and "See Also" sections

**1.3 Verify trait modules**
```bash
for file in docs/api/traits/*.md; do
    echo "$file:"; grep -n "^#### \|^### " "$file" | head -2
done
```
Expected: No manual headings before ::: blocks

**Tasks to Complete**:
- [ ] Document all manual headings in series.md
- [ ] Document all manual headings in pipelines.md
- [ ] Confirm trait modules follow correct pattern
- [ ] Create audit report

---

### Phase 2: Fix series.md (20-30 min)

**2.1 Remove manual #### headings**

Expected headings to remove:
- `#### load` (line ~57, before Series.load block)
- `#### get_primary_points` (line ~129, before get_primary_points block)
- `#### get_lateral_points` (line ~165, before get_lateral_points block)
- `#### get_crown_points` (line ~218, before get_crown_points block)
- Additional methods (identified in audit)

**Regex Pattern**:
```
^#### (\w+)\n\n(?=::: sleap_roots\.Series\.)
```
Replace with: (empty - delete)

**2.2 Fix bullet lists**

For each "See Also" section:
1. Add blank line before bullet list
2. Ensure proper positioning after example blocks
3. Verify list formatting

**Tasks to Complete**:
- [ ] Remove all manual #### headings
- [ ] Add blank lines before bullet lists
- [ ] Verify all "See Also" positioned correctly
- [ ] Check file structure integrity

---

### Phase 3: Fix pipelines.md (20-30 min)

**3.1 Remove conflicting ## headings**

These manual headings conflict with mkdocstrings auto-generation:
- `## DicotPipeline` (before mkdocstrings block with heading_level: 3)
- `## MultipleDicotPipeline`
- `## YoungerMonocotPipeline`
- `## OlderMonocotPipeline`
- `## PrimaryRootPipeline`
- `## MultiplePrimaryRootPipeline`
- `## LateralRootPipeline`

**Important**: Keep subsection headings like `### Overview`, `### Computed Traits` (not duplicates)

**Regex Pattern**:
```
^## (DicotPipeline|MultipleDicotPipeline|YoungerMonocotPipeline|OlderMonocotPipeline|PrimaryRootPipeline|MultiplePrimaryRootPipeline|LateralRootPipeline)\n\n
```
Replace with: (empty - delete)

**3.2 Fix bullet lists**

Same as series.md - add blank lines before bullet lists

**Tasks to Complete**:
- [ ] Remove all conflicting manual headings
- [ ] Verify subsections preserved
- [ ] Fix bullet list formatting
- [ ] Verify structure integrity

---

### Phase 4: Trait Module Verification (10 min)

**4.1 Quick scan all trait modules**
```bash
grep -n "^#### \|^### " docs/api/traits/*.md
```

**4.2 Verify pattern**

For each trait module:
- No manual #### or ### before ::: blocks
- "See Also" sections properly formatted
- Blank lines before bullet lists

**Expected Result**: All trait modules follow correct pattern - no changes needed

**Tasks to Complete**:
- [ ] Scan all trait modules
- [ ] Document pattern verification
- [ ] Confirm no changes needed

---

### Phase 5: Test Build (15-20 min)

**5.1 Build documentation**
```bash
cd /Users/elizabethberrigan/repos/sleap-roots
mkdocs build --strict
```

**Success Criteria**:
- [ ] Build completes without errors
- [ ] No warnings about duplicate headings
- [ ] No rendering errors
- [ ] HTML generated successfully

**5.2 Check build output**
```bash
ls -la site/api/core/series/
ls -la site/api/core/pipelines/
```

**5.3 Visual inspection** (if possible)
- Open generated HTML in browser
- Verify no duplicate headings
- Verify bullet lists render as bullets
- Check anchor links function

**5.4 Git diff review**
```bash
git diff --stat
git diff docs/api/core/series.md | head -50
git diff docs/api/core/pipelines.md | head -50
```

**Tasks to Complete**:
- [ ] Run mkdocs build
- [ ] Verify build success
- [ ] Visual inspection
- [ ] Review git diffs

---

### Phase 6: Commit Changes (10 min)

**6.1 Commit series.md**
```bash
git add docs/api/core/series.md
git commit -m "fix: remove duplicate headings in series.md API docs

- Remove manual #### headings before mkdocstrings blocks
- mkdocstrings heading_level: 4 now handles heading generation
- Add proper spacing before bullet lists in 'See Also' sections
- Ensures single, clean headings in rendered output"
```

**6.2 Commit pipelines.md**
```bash
git add docs/api/core/pipelines.md
git commit -m "fix: remove duplicate headings in pipelines.md API docs

- Remove manual ### headings before mkdocstrings blocks
- Keep ### subsections like Overview and Computed Traits
- mkdocstrings heading_level: 3 now handles heading generation
- Improves documentation consistency and rendering"
```

**6.3 Final verification**
```bash
git status
git log --oneline -5
```

**Tasks to Complete**:
- [ ] Review and commit series.md changes
- [ ] Review and commit pipelines.md changes
- [ ] Verify clean git status
- [ ] Verify commits in log

---

## TIMELINE & CHECKPOINTS

```
┌─ Phase 1: Audit (15 min)
│  └─ Checkpoint: All manual headings documented
│
├─ Phase 2: series.md fixes (25 min)
│  └─ Checkpoint: ~8 headings removed, lists formatted
│
├─ Phase 3: pipelines.md fixes (25 min)
│  └─ Checkpoint: ~7 headings removed, structure intact
│
├─ Phase 4: Trait module verification (10 min)
│  └─ Checkpoint: Pattern verified, no changes needed
│
├─ Phase 5: Build & test (20 min)
│  └─ Checkpoint: Build succeeds, visual inspection complete
│
└─ Phase 6: Commit (10 min)
   └─ Checkpoint: Clean git status, commits recorded

TOTAL: 1.5-2 hours
```

---

## SUCCESS CRITERIA

All must be satisfied:

- [x] Analysis complete and documented
- [ ] No duplicate headings in rendered API docs
- [ ] All "See Also" bullet lists render correctly
- [ ] mkdocs build completes without errors
- [ ] No warnings about rendering or structure
- [ ] Visual inspection confirms proper formatting
- [ ] Git history shows intentional, well-documented changes
- [ ] Documentation consistency improved across all API docs

---

## KEY COMMANDS REFERENCE

```bash
# Audit
grep -n "^#### " docs/api/core/series.md
grep -n "^## \|^### " docs/api/core/pipelines.md
grep -n "See Also" docs/api/core/*.md

# Build
mkdocs build --strict

# Git operations
git diff docs/api/core/series.md
git diff docs/api/core/pipelines.md
git log --oneline -5

# Verification
grep -n "^#### \|^### " docs/api/traits/*.md
```

---

## ROLLBACK PROCEDURES

**If build fails**:
```bash
git revert <commit-hash>
```

**If specific file needs restoration**:
```bash
git checkout HEAD -- docs/api/core/series.md
```

**If mkdocstrings behavior unexpected**:
- Check mkdocs.yml configuration
- Verify heading_level settings
- Review mkdocstrings documentation

---

## NEXT ACTION

When ready to start execution:
1. Begin Phase 1 (Comprehensive Audit)
2. Document findings
3. Proceed with Phases 2-3 based on audit results
4. Complete test and commit phases
5. Document completion in memory

Plan is fully analyzed, documented, and ready for systematic implementation.
