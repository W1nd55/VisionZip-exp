#!/bin/bash
# cleanup_repo.sh
# Organize experimental files into experimentations/ folder

set -e

echo "=========================================="
echo "Repository Cleanup Script"
echo "=========================================="
echo ""

# Create experimentations folder
mkdir -p experimentations

echo "Moving experimental scripts..."

# Move fix scripts
mv -v fix_all.sh experimentations/ 2>/dev/null || true
mv -v fix_generation.py experimentations/ 2>/dev/null || true
mv -v fix_generation_final.py experimentations/ 2>/dev/null || true
mv -v fix_mme_structure.sh experimentations/ 2>/dev/null || true
mv -v fix_path.sh experimentations/ 2>/dev/null || true
mv -v fix_paths_and_run.sh experimentations/ 2>/dev/null || true
mv -v fix_syntax.py experimentations/ 2>/dev/null || true

# Move setup/update scripts
mv -v setup_vm.sh experimentations/ 2>/dev/null || true
mv -v update_vm.py experimentations/ 2>/dev/null || true
mv -v start.sh experimentations/ 2>/dev/null || true

# Move documentation that's experimental
mv -v MODIFICATIONS_SUMMARY.md experimentations/ 2>/dev/null || true
mv -v EVALUATION_RESULTS.md experimentations/ 2>/dev/null || true

# Move summary CSV from parent if it exists
if [ -f ../summary.csv ]; then
    mv -v ../summary.csv experimentations/ 2>/dev/null || true
    echo "Moved summary.csv from parent directory"
fi

echo ""
echo "Removing tar files..."
rm -vf *.tar.gz 2>/dev/null || true

echo ""
echo "Creating README for experimentations folder..."
cat > experimentations/README.md << 'EOF'
# Experimentations

This folder contains scripts and documentation from the development and debugging process.

## Fix Scripts
- `fix_*.sh/py`: Various scripts used to fix bugs and issues during development
- `setup_vm.sh`: VM setup automation
- `update_vm.py`: VM update helper

## Documentation
- `MODIFICATIONS_SUMMARY.md`: Detailed log of all code changes
- `EVALUATION_RESULTS.md`: Initial evaluation results and analysis
- `summary.csv`: Aggregated benchmark results

## Note
These files are kept for reference but are not required for running the codebase.
EOF

echo ""
echo "=========================================="
echo "Cleanup Complete!"
echo "=========================================="
echo ""
echo "Moved to experimentations/:"
ls -1 experimentations/ | grep -v README.md || echo "  (none)"
echo ""
echo "Current root directory is now clean."
echo "Essential files kept:"
echo "  - run_mme_eval.sh"
echo "  - run_full_evaluation.sh"
echo "  - run_cross_attention_eval.sh"
echo "  - activate_env.sh"
echo "  - sparsezip_flowchart.png"
echo ""
