#!/bin/bash
# Automatically runs Phases 2-3 once Phase 1 completes
source /Users/qjy/anaconda3/bin/activate cs572

cd "/Users/qjy/Emory/2026spring/CS 572/cs572-finalProject/router_v2"

echo "[$(date '+%H:%M:%S')] Waiting for Phase 1 to complete..."

# Wait for labels_train.json to have substantial content
while true; do
  if [ -f labels_train.json ]; then
    count=$(grep -c ":" labels_train.json 2>/dev/null || echo 0)
    if [ "$count" -gt 1000 ]; then
      echo "[$(date '+%H:%M:%S')] Phase 1 complete ($count labels)"
      break
    fi
  fi
  sleep 30
done

echo ""
echo "========================================"
echo "✓ Phase 1 COMPLETE"
echo "========================================"
echo ""

echo "[$(date '+%H:%M:%S')] Starting Phase 2: Feature extraction..."
python extract_features_sbert.py | tee -a phase2.log

if [ $? -eq 0 ]; then
  echo ""
  echo "========================================"
  echo "✓ Phase 2 COMPLETE"
  echo "========================================"
  echo ""

  echo "[$(date '+%H:%M:%S')] Starting Phase 3: Model training..."
  python train.py | tee -a phase3.log

  echo ""
  echo "========================================"
  echo "✓ Phase 3 COMPLETE"
  echo "========================================"
  echo ""
  
  if [ -f model.pkl ] && [ -f threshold.json ]; then
    echo "✅ FULL PIPELINE COMPLETE!"
    echo ""
    ls -lh model.pkl threshold.json labels_train.json
  fi
else
  echo "ERROR: Phase 2 failed"
  exit 1
fi
