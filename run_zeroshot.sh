#!/bin/bash


# ============================================================================
# Configuration
# ============================================================================

# Required: Set these paths
TEST_FILE="     "  # Path to the test file listing video IDs and labels
VIDEOS_ROOT="     "    # Root directory containing video files
OUTPUT_BASE_DIR="./experiments_output" # Base directory to save experiment outputs
GPU_ID="0"


# Prompt flags to test
PROMPT_FLAGS="s d1 d2 td1 td2"

# ============================================================================
# Check Prerequisites
# ============================================================================

echo "============================================"
echo "Video Ambivalence Classification Experiments"
echo "============================================"
echo ""

# Check if test file exists
if [ ! -f "$TEST_FILE" ]; then
    echo "Error: Test file not found: $TEST_FILE"
    echo "Please set the TEST_FILE variable in this script."
    exit 1
fi

# Check if videos root exists
if [ ! -d "$VIDEOS_ROOT" ]; then
    echo "Error: Videos root directory not found: $VIDEOS_ROOT"
    echo "Please set the VIDEOS_ROOT variable in this script."
    exit 1
fi

# Check if Python script exists
if [ ! -f "video_ambivalence_classifier.py" ]; then
    echo "Error: video_ambivalence_classifier.py not found in current directory"
    exit 1
fi

# Create base output directory
mkdir -p "$OUTPUT_BASE_DIR"

echo "Configuration:"
echo "  Test File: $TEST_FILE"
echo "  Videos Root: $VIDEOS_ROOT"
echo "  Output Directory: $OUTPUT_BASE_DIR"
echo "  GPU ID: $GPU_ID"
echo ""

# ============================================================================
# Run Experiments
# ============================================================================

echo "Starting experiments..."
echo ""

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_BASE_DIR/experiment_log_$TIMESTAMP.txt"

# Redirect all output to both console and log file
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "Experiment started at: $(date)"
echo "============================================"
echo ""

# Run for each prompt flag
for PROMPT_FLAG in $PROMPT_FLAGS; do
    echo "-------------------------------------------"
    echo "Running experiment with prompt: $PROMPT_FLAG"
    echo "-------------------------------------------"
    
    OUTPUT_DIR="$OUTPUT_BASE_DIR/prompt_$PROMPT_FLAG"
    
    # Run the classification
    START_TIME=$(date +%s)
    
    python video_ambivalence_classifier.py \
        --test_file "$TEST_FILE" \
        --videos_root "$VIDEOS_ROOT" \
        --prompt_flag "$PROMPT_FLAG" \
        --output_dir "$OUTPUT_DIR" \
        --gpu "$GPU_ID"
    
    EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "SUCCESS - Completed in ${DURATION}s"
    else
        echo "FAILED - Exit code $EXIT_CODE"
    fi
    
    echo ""
done

echo "============================================"
echo "All experiments completed at: $(date)"
echo ""

# ============================================================================
# Compile Results from CSV Files
# ============================================================================

echo "============================================"
echo "Compiling Results"
echo "============================================"
echo ""

RESULTS_FILE="$OUTPUT_BASE_DIR/compiled_results_$TIMESTAMP.csv"
SUMMARY_FILE="$OUTPUT_BASE_DIR/results_summary_$TIMESTAMP.txt"

# Create Python script to calculate metrics from CSV files
METRICS_SCRIPT="$OUTPUT_BASE_DIR/calculate_metrics.py"

cat > "$METRICS_SCRIPT" <<'PYTHON_EOF'
import pandas as pd
import sys
from sklearn.metrics import f1_score, accuracy_score

def calculate_metrics(predictions_file, labels_file):
    try:
        pred_df = pd.read_csv(predictions_file)
        label_df = pd.read_csv(labels_file)
        
        predictions = pred_df['response'].values
        labels = label_df['label'].values
        
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='macro')
        
        print(f"{accuracy:.4f},{f1:.4f}")
        return 0
    except Exception as e:
        print(f"ERROR,ERROR")
        return 1

if __name__ == "__main__":
    predictions_file = sys.argv[1]
    labels_file = sys.argv[2]
    sys.exit(calculate_metrics(predictions_file, labels_file))
PYTHON_EOF

echo "Calculating metrics from prediction files..."
echo ""

# Create header for CSV
echo "Prompt,Description,Accuracy,F1_Score,Status" > "$RESULTS_FILE"

# Extract metrics from each experiment
for PROMPT_FLAG in $PROMPT_FLAGS; do
    OUTPUT_DIR="$OUTPUT_BASE_DIR/prompt_$PROMPT_FLAG"
    
    # Get prompt description
    case "$PROMPT_FLAG" in
        s)   DESC="Simple - No context" ;;
        d1)  DESC="Definition 1 - Contradictory feelings" ;;
        d2)  DESC="Definition 2 - Desires for and against change" ;;
        td1) DESC="Transcript plus Definition 1" ;;
        td2) DESC="Transcript plus Definition 2" ;;
        *)   DESC="Unknown" ;;
    esac
    
    # Check if prediction files exist
    PRED_FILE="$OUTPUT_DIR/predictions_${PROMPT_FLAG}.csv"
    LABEL_FILE="$OUTPUT_DIR/ground_truth_labels.csv"
    
    if [ -f "$PRED_FILE" ] && [ -f "$LABEL_FILE" ]; then
        # Calculate metrics using Python
        METRICS=$(python "$METRICS_SCRIPT" "$PRED_FILE" "$LABEL_FILE")
        
        if [[ "$METRICS" != "ERROR,ERROR" ]]; then
            ACCURACY=$(echo "$METRICS" | cut -d',' -f1)
            F1=$(echo "$METRICS" | cut -d',' -f2)
            
            echo "$PROMPT_FLAG,$DESC,$ACCURACY,$F1,Success" >> "$RESULTS_FILE"
            echo "[$PROMPT_FLAG] $DESC"
            echo "  Accuracy: $ACCURACY"
            echo "  F1 Score: $F1"
        else
            echo "$PROMPT_FLAG,$DESC,N/A,N/A,Error calculating metrics" >> "$RESULTS_FILE"
            echo "[$PROMPT_FLAG] $DESC"
            echo "  Error calculating metrics"
        fi
    else
        echo "$PROMPT_FLAG,$DESC,N/A,N/A,Files not found" >> "$RESULTS_FILE"
        echo "[$PROMPT_FLAG] $DESC"
        echo "  Prediction files not found"
    fi
    echo ""
done

# Clean up temporary script
rm -f "$METRICS_SCRIPT"

# ============================================================================
# Create Summary Report
# ============================================================================

echo "Creating summary report..."
echo ""

cat > "$SUMMARY_FILE" <<EOF
============================================
Video Ambivalence Classification Results
============================================

Experiment Timestamp: $TIMESTAMP
Test File: $TEST_FILE
Videos Root: $VIDEOS_ROOT

--------------------------------------------
Results Summary
--------------------------------------------

EOF

# Add table header
echo "Prompt | Description                              | Accuracy | F1 Score | Status" >> "$SUMMARY_FILE"
echo "-------|------------------------------------------|----------|----------|--------" >> "$SUMMARY_FILE"

# Read results from CSV and format table
tail -n +2 "$RESULTS_FILE" | while IFS=',' read -r prompt desc accuracy f1 status; do
    printf "%-6s | %-40s | %8s | %8s | %s\n" "$prompt" "$desc" "$accuracy" "$f1" "$status" >> "$SUMMARY_FILE"
done

cat >> "$SUMMARY_FILE" <<EOF

--------------------------------------------
Output Files
--------------------------------------------

Results CSV: $RESULTS_FILE
Full Log: $LOG_FILE

Individual experiment outputs:
EOF

for PROMPT_FLAG in $PROMPT_FLAGS; do
    echo "  - $OUTPUT_BASE_DIR/prompt_$PROMPT_FLAG/" >> "$SUMMARY_FILE"
done

cat >> "$SUMMARY_FILE" <<EOF

--------------------------------------------
Prompt Descriptions
--------------------------------------------

s   : Simple prompt without additional context
d1  : Includes definition of ambivalence as contradictory feelings
d2  : Includes definition focusing on change desires
td1 : Combines transcript with definition 1
td2 : Combines transcript with definition 2

============================================
EOF

# Display summary
echo "============================================"
echo "Experiment Summary"
echo "============================================"
echo ""
cat "$SUMMARY_FILE"
echo ""

echo "Files saved:"
echo "  - Results CSV: $RESULTS_FILE"
echo "  - Summary: $SUMMARY_FILE"
echo "  - Full log: $LOG_FILE"
echo ""
echo "All experiments completed!"