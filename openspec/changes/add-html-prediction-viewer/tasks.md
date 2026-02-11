## 1. Module Structure Setup
- [ ] 1.1 Create `sleap_roots/viewer/` directory structure
- [ ] 1.2 Create `__init__.py` with public API exports
- [ ] 1.3 Add viewer CLI entry point to `pyproject.toml`

## 2. Core Rendering (TDD)
- [ ] 2.1 Write tests for frame rendering with root type overlay
- [ ] 2.2 Implement root type rendering using Series.plot()
- [ ] 2.3 Write tests for frame rendering with confidence colormap overlay
- [ ] 2.4 Implement confidence colormap rendering (viridis mapped to score range)
- [ ] 2.5 Write tests for base64 image encoding
- [ ] 2.6 Implement base64 encoding utility

## 3. Scan Discovery (TDD)
- [ ] 3.1 Write tests for scan discovery from predictions directory
- [ ] 3.2 Implement `ScanInfo` dataclass
- [ ] 3.3 Implement scan discovery logic in `generator.py`
- [ ] 3.4 Write tests for handling missing/invalid files

## 4. HTML Generation (TDD)
- [ ] 4.1 Write tests for HTML template rendering
- [ ] 4.2 Create Jinja2 template `templates/viewer.html`
- [ ] 4.3 Implement `ViewerGenerator.generate()` method
- [ ] 4.4 Write tests for complete generation workflow

## 5. CLI Implementation
- [ ] 5.1 Write tests for CLI argument parsing
- [ ] 5.2 Implement `cli.py` with Click commands
- [ ] 5.3 Add progress bar for generation
- [ ] 5.4 Write integration test for full CLI workflow

## 6. JavaScript Navigation
- [ ] 6.1 Implement keyboard navigation (arrows, Enter, Esc)
- [ ] 6.2 Implement scan overview grid view
- [ ] 6.3 Implement frame drill-down view
- [ ] 6.4 Add frame counter and navigation controls
- [ ] 6.5 Implement toggle between root type and confidence views (key 'C' or button)
- [ ] 6.6 Add color legend for confidence colormap view
- [ ] 6.7 Add mode indicator showing current view (root type / confidence)

## 7. Quality & Polish
- [ ] 7.1 Add confidence score display
- [ ] 7.2 Add instance count display
- [ ] 7.3 Ensure responsive CSS layout
- [ ] 7.4 Test with primary-only, dicot, and monocot predictions

## 8. Documentation
- [ ] 8.1 Create `docs/guides/viewer.md` user guide
- [ ] 8.2 Add docstrings to all public functions
- [ ] 8.3 Update README with viewer feature

## 9. Final Validation
- [ ] 9.1 Run full test suite (`uv run pytest`)
- [ ] 9.2 Run linting (`uv run black --check`, `uv run pydocstyle`)
- [ ] 9.3 Test on real experiment data
- [ ] 9.4 Verify HTML output in multiple browsers