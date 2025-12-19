#!/bin/bash

# Script to update headers in thesis documentation
# H1 → "Chương X: Title"
# H2 → "X.Y. Title"
# H3 → "X.Y.Z. Title"

update_h1() {
    local file="$1"
    local chapter="$2"

    # Get current H1 line
    local h1_line=$(grep -m1 "^# " "$file")

    # Skip if already has "Chương"
    if [[ "$h1_line" == *"Chương"* ]]; then
        echo "  H1 already has Chương prefix, skipping"
        return
    fi

    # Extract title (remove "# " prefix)
    local title="${h1_line:2}"

    # Create new H1
    local new_h1="# Chương $chapter: $title"

    # Replace first H1 only
    sed -i "0,/^# /{s|^# .*|$new_h1|}" "$file"
    echo "  H1 updated to: $new_h1"
}

update_h2_numbering() {
    local file="$1"
    local chapter="$2"
    local section="$3"

    # This is complex - H2 needs to be X.Y. format
    # For now, we'll handle this case by case
    echo "  H2 numbering: $chapter.$section.X format"
}

# Process each chapter
echo "=== Updating Headers ==="

# Chapter 1
for file in research/chuong-01-gioi-thieu/muc-*/*.md; do
    echo "Processing: $file"
    update_h1 "$file" "1"
done

# Chapter 2
for file in research/chuong-02-co-so-ly-thuyet/muc-*/*.md; do
    echo "Processing: $file"
    update_h1 "$file" "2"
done

# Chapter 3
for file in research/chuong-03-phat-hien-tau-bien/muc-*/*.md; do
    echo "Processing: $file"
    update_h1 "$file" "3"
done

# Chapter 4
for file in research/chuong-04-phat-hien-dau-loang/muc-*/*.md; do
    echo "Processing: $file"
    update_h1 "$file" "4"
done

# Chapter 5
for file in research/chuong-05-torchgeo/muc-*/*.md; do
    echo "Processing: $file"
    update_h1 "$file" "5"
done

# Chapter 6
for file in research/chuong-06-xview-challenges/muc-*/*.md; do
    echo "Processing: $file"
    update_h1 "$file" "6"
done

# Chapter 7
for file in research/chuong-07-ket-luan/muc-*/*.md; do
    echo "Processing: $file"
    update_h1 "$file" "7"
done

echo "=== Done updating H1 headers ==="
