#!/bin/bash
source /Users/qjy/anaconda3/bin/activate cs572

echo "Waiting for Phase 2 to complete..."

# Wait for feature files to exist
while true; do
  if [ -f features_sbert_train.npz ] && [ -f features_sbert_validation.npz ]; then
    echo "Phase 2 complete - feature files found"
    break
  fi
  sleep 10
done

sleep 2
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

if [ -f model.pkl ] && [ -f threshold.json ]; then
  echo "✅ FULL PIPELINE COMPLETE!"
  echo ""
  ls -lh model.pkl threshold.json
fi
