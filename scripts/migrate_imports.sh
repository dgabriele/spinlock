#!/bin/bash
# Migration script for updating experiment imports

set -e

echo "Starting import migration..."

# Update Python files in src/spinlock/experimental/
echo "Updating imports in src/spinlock/experimental/..."
find src/spinlock/experimental -name "*.py" -type f -exec sed -i \
  's/from experiments\./from spinlock.experimental./g' {} +

find src/spinlock/experimental -name "*.py" -type f -exec sed -i \
  's/import experiments\./import spinlock.experimental./g' {} +

# Update CLI command
echo "Updating CLI command imports..."
sed -i 's/from experiments\./from spinlock.experimental./g' \
  src/spinlock/cli/visualize_diffusion_inpainting.py

sed -i 's/import experiments\./import spinlock.experimental./g' \
  src/spinlock/cli/visualize_diffusion_inpainting.py

# Update standalone scripts in experiments/
echo "Updating standalone scripts..."
find experiments -name "*.py" -type f -exec sed -i \
  's/from experiments\./from spinlock.experimental./g' {} +

find experiments -name "*.py" -type f -exec sed -i \
  's/import experiments\./import spinlock.experimental./g' {} +

echo "Import migration complete!"
echo ""
echo "Verifying no leftover 'from experiments.' imports in src/spinlock/..."
if grep -r "from experiments\." src/spinlock/ 2>/dev/null; then
    echo "❌ Found leftover imports"
    exit 1
else
    echo "✓ All imports updated in src/spinlock/"
fi

echo ""
echo "Checking for leftover 'import experiments.' imports in src/spinlock/..."
if grep -r "import experiments\." src/spinlock/ 2>/dev/null; then
    echo "❌ Found leftover imports"
    exit 1
else
    echo "✓ All imports updated in src/spinlock/"
fi

echo ""
echo "✓ Migration successful!"
