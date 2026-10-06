# Windows installer

Run install.bat from a PowerShell or Command Prompt. The installer copies the
source to %LOCALAPPDATA%\Palmistry, creates a Python virtual environment,
installs the base requirements (including ZIP/RAR import support), creates Start Menu/Desktop shortcuts, and
launches the native PySide6 desktop window. The shortcut uses the embedded Qt WebEngine interface; it does not open Opera or the default browser.

`-WithModel` is retained for compatibility; the NumPy model has no optional ML dependencies. Use
install.bat -SkipDependencies when installing from an offline wheel cache.

The installer copies pipeline.py and bootstrap_review.py too. The app exposes ZIP/RAR import, review queue and bootstrap/train actions in its UI. CLI files remain available for advanced automation. A RAR archive also needs a local 7-Zip, unrar or bsdtar executable. The UI uploads large archives in 16 MB chunks and processes extraction in a background job; RAR files are extracted through one batch backend process and the UI reports validation progress. Keep at least the archive size plus room for the extracted images free on the target drive. The bootstrap script trains from approved masks, keeps only confidence-thresholded labels for the remainder, and records skipped rows and provenance.
