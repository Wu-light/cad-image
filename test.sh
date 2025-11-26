#!/bin/bash
# Test script for BrepVis - Comprehensive feature demonstration
# Usage: ./test.sh

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test configuration
TEST_FILE="samples/example.step"
OUTPUT_DIR="test_output"

# Check if test file exists
if [ ! -f "$TEST_FILE" ]; then
    echo -e "${RED}Error: Test file $TEST_FILE not found${NC}"
    echo "Please ensure samples/example.step exists"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║    BrepVis Comprehensive Test Suite   ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""
echo "Test file: $TEST_FILE"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Function to run render command tests
run_test() {
    local test_name="$1"
    shift
    echo -e "${BLUE}► $test_name${NC}"
    echo "  Command: python vis_step.py render $@"
    if python vis_step.py render "$@" > /dev/null 2>&1; then
        echo -e "${GREEN}  ✓ Passed${NC}"
        echo ""
    else
        echo -e "${RED}  ✗ Failed${NC}"
        echo ""
        exit 1
    fi
}

# Clean output directory
echo -e "${YELLOW}Cleaning previous outputs...${NC}"
rm -rf "$OUTPUT_DIR"/*
echo ""

# ============================================================================
# Section A: Core Rendering
# ============================================================================

echo -e "${BLUE}═══ A. Core Rendering ═══${NC}"
echo ""

run_test "1. Basic rendering (default settings)" \
    "$TEST_FILE" --output-dir "$OUTPUT_DIR" --output-name "01_basic.png"

run_test "2. High resolution (512x512)" \
    "$TEST_FILE" --output-dir "$OUTPUT_DIR" --output-name "02_highres.png" \
    --resolution 512

run_test "3. Color variations (pink)" \
    "$TEST_FILE" --output-dir "$OUTPUT_DIR" --output-name "03_color_pink.png" \
    --color pink

run_test "4. Transparent background (RGBA mode)" \
    "$TEST_FILE" --output-dir "$OUTPUT_DIR" --output-name "04_rgba.png" \
    --color-mode rgba

# ============================================================================
# Section B: Transformations
# ============================================================================

echo -e "${BLUE}═══ B. Transformations ═══${NC}"
echo ""

run_test "5. Stand upright (90° X rotation)" \
    "$TEST_FILE" --output-dir "$OUTPUT_DIR" --output-name "05_upright.png" \
    --stand-upright

run_test "6. Flip Z (mirror XZ plane)" \
    "$TEST_FILE" --output-dir "$OUTPUT_DIR" --output-name "06_flip_z.png" \
    --flip-z

run_test "7. Combined transformations" \
    "$TEST_FILE" --output-dir "$OUTPUT_DIR" --output-name "07_combined.png" \
    --stand-upright --flip-z

# ============================================================================
# Section C: More Features
# ============================================================================

echo -e "${BLUE}═══ C. More Features ═══${NC}"
echo ""

run_test "8. Partial face rendering (faces 0,1)" \
    "$TEST_FILE" --output-dir "$OUTPUT_DIR" --output-name "08_partial.png" \
    --partial "0,1"

run_test "9. Fast mode (coarse mesh)" \
    "$TEST_FILE" --output-dir "$OUTPUT_DIR" --output-name "09_fast.png" \
    --fast

run_test "10. Camera: top-down view" \
    "$TEST_FILE" --output-dir "$OUTPUT_DIR" --output-name "10_camera_topdown.png" \
    --camera-base-angle -90 --camera-height 3.0

run_test "11. Camera: close-up view" \
    "$TEST_FILE" --output-dir "$OUTPUT_DIR" --output-name "11_camera_closeup.png" \
    --camera-distance 2.0 --camera-height 1.5

run_test "12. Ground plane at z=0.0" \
    "$TEST_FILE" --output-dir "$OUTPUT_DIR" --output-name "12_ground_z0.png" \
    --ground-plane-z 0.0

run_test "13. Exploded view (faces/edges)" \
    "$TEST_FILE" --output-dir "$OUTPUT_DIR" --explode --fast --camera-distance 4.0

# ============================================================================
# Section D: Video Rendering
# ============================================================================

echo -e "${BLUE}═══ D. Video Rendering ═══${NC}"
echo ""

if command -v ffmpeg &> /dev/null; then
    echo -e "${BLUE}► 14. MP4 video (2s, 24 FPS)${NC}"
    echo "  Command: python vis_step.py render-video $TEST_FILE --output-path $OUTPUT_DIR/14_video.mp4 --duration 2.0 --fps 24 --resolution 256"
    if python vis_step.py render-video "$TEST_FILE" \
        --output-path "$OUTPUT_DIR/14_video.mp4" \
        --duration 2.0 --fps 24 --resolution 256 > /dev/null 2>&1; then
        echo -e "${GREEN}  ✓ Passed${NC}"
        echo ""
    else
        echo -e "${RED}  ✗ Failed${NC}"
        echo ""
        exit 1
    fi

    echo -e "${BLUE}► 15. GIF with custom camera${NC}"
    echo "  Command: python vis_step.py render-video $TEST_FILE --output-path $OUTPUT_DIR/15_video_camera.gif --duration 2.0 --fps 24 --resolution 256 --video-format gif --camera-base-angle -60 --camera-distance 2.5"
    if python vis_step.py render-video "$TEST_FILE" \
        --output-path "$OUTPUT_DIR/15_video_camera.gif" \
        --duration 2.0 --fps 24 --resolution 256 \
        --video-format gif \
        --camera-base-angle -60 --camera-distance 2.5 > /dev/null 2>&1; then
        echo -e "${GREEN}  ✓ Passed${NC}"
        echo ""
    else
        echo -e "${RED}  ✗ Failed${NC}"
        echo ""
        exit 1
    fi
else
    echo -e "${YELLOW}  ⊘ Skipping video tests (ffmpeg not installed)${NC}"
    echo ""
fi

# ============================================================================
# Section E: Interactive Viewer (Optional)
# ============================================================================

echo -e "${BLUE}═══ E. Interactive Viewer (Optional) ═══${NC}"
echo ""
echo -e "${YELLOW}Press Enter to launch Polyscope viewer, or Ctrl+C to skip...${NC}"
read -r

if python -c "import polyscope" 2>/dev/null; then
    echo -e "${BLUE}► Launching Polyscope viewer...${NC}"
    echo "  Command: python vis_step.py view $TEST_FILE"
    python vis_step.py view "$TEST_FILE" || true
    echo -e "${GREEN}  ✓ Viewer closed${NC}"
    echo ""
else
    echo -e "${YELLOW}  ⊘ Polyscope not installed (pip install polyscope)${NC}"
    echo ""
fi

# ============================================================================
# Summary
# ============================================================================

echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║       All Tests Completed! ✓           ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
echo ""
echo "Output directory: $OUTPUT_DIR/"
echo ""
echo "Features tested:"
echo -e "${GREEN}  ✓${NC} Core rendering (default, high-res, colors, RGBA)"
echo -e "${GREEN}  ✓${NC} Transformations (stand-upright, flip-z, combined)"
echo -e "${GREEN}  ✓${NC} Camera control (distance, height, base angle)"
echo -e "${GREEN}  ✓${NC} Ground plane configuration (custom z position)"
echo -e "${GREEN}  ✓${NC} More: partial faces, fast mode, exploded view"
if command -v ffmpeg &> /dev/null; then
    echo -e "${GREEN}  ✓${NC} Video rendering (MP4, GIF with camera control)"
else
    echo -e "${YELLOW}  ⊘${NC} Video rendering (ffmpeg not installed)"
fi
if python -c "import polyscope" 2>/dev/null; then
    echo -e "${GREEN}  ✓${NC} Interactive viewer (Polyscope)"
else
    echo -e "${YELLOW}  ⊘${NC} Interactive viewer (polyscope not installed)"
fi
echo ""
echo "Example outputs:"
echo "  • Basic renders: $OUTPUT_DIR/01_basic.png through 13_*.png"
echo "  • Exploded view: $OUTPUT_DIR/exploded/face_*.png"
if command -v ffmpeg &> /dev/null; then
    echo "  • Videos: $OUTPUT_DIR/14_video.mp4, 15_video_camera.gif"
fi
echo ""
