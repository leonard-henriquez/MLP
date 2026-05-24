#!/bin/sh
# Download the MNIST database into data/mnist (or the given directory).
set -e

DIR=${1:-data/mnist}
FILES="train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte"
# yann.lecun.com no longer serves the files: try a mirror first.
MIRRORS="https://ossci-datasets.s3.amazonaws.com/mnist http://yann.lecun.com/exdb/mnist"

mkdir -p "$DIR"
# Keep the downloaded files out of Git.
echo "*" > "$DIR/.gitignore"
for file in $FILES; do
    if [ -f "$DIR/$file" ]; then
        echo "$DIR/$file already there"
        continue
    fi
    for mirror in $MIRRORS; do
        echo "Downloading $mirror/$file.gz"
        if curl -fsSL --connect-timeout 10 -o "$DIR/$file.gz" "$mirror/$file.gz" 2>/dev/null; then
            break
        fi
        rm -f "$DIR/$file.gz"
        echo "  not available, trying the next mirror"
    done
    if [ ! -f "$DIR/$file.gz" ]; then
        echo "Could not download $file" >&2
        exit 1
    fi
    gunzip "$DIR/$file.gz"
done
echo "MNIST is ready in $DIR"
