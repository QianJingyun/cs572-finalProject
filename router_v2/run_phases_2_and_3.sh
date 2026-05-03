#!/bin/bash
# Runs Phase 2 and Phase 3 once Phase 1 completes

echo "Waiting for Phase 1 (label generation) to complete..."
echo "This can take 1-2 hours..."

# Wait for Phase 1 to complete (check if labels_train.json exists and has content)
until [ -f labels_train.json ] && [ $(wc -l < labels_train.json) -gt 100 ]; do
  count=$(grep -c ":" labels_train.json 2>/dev/null || echo 0)
  echo "[$(date '+%H:%M:%S')] Phase 1 progress: $count labels generated"
  sleep 30
done

echo ""
echo "========================================"
echo "✓ Phase 1 COMPLETE"
echo "========================================"
echo ""

echo "Starting Phase 2: Feature extraction..."
python extract_features_sbert.py | tee phase2.log

echo ""
echo "========================================"
echo "✓ Phase 2 COMPLETE"
echo "========================================"
echo ""

echo "Starting Phase 3: Model training..."
python train.py | tee phase3.log

echo ""
echo "========================================"
echo "✓ Phase 3 COMPLETE"
echo "========================================"
echo ""

echo "All phases completed!"
ls -lh model.pkl threshold.json
