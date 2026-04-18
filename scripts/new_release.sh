#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:-}"
DATE=$(date +%F)

if [[ -z "${VERSION:-}" ]]; then
  echo "Usage: ./release.sh <version>"
  exit 1
fi

echo "Releasing version: $VERSION"
echo "Date: $DATE"

############################
# 1. pyproject.toml
############################
if [[ -f pyproject.toml ]]; then
  sed -i.bak "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml
  echo "Updated pyproject.toml"
fi

############################
# 2. CITATION.cff
############################
if [[ -f CITATION.cff ]]; then
  sed -i.bak "s/^version: .*/version: $VERSION/" CITATION.cff
  sed -i.bak "s/^date-released: .*/date-released: $DATE/" CITATION.cff
  echo "Updated CITATION.cff"
fi

############################
# 3. Dockerfile
############################
if [[ -f binder/Dockerfile ]]; then
  sed -i.bak -E "s|(FROM ghcr.io/plant-food-research-open/calisim:)[^ ]+|\1$VERSION|" binder/Dockerfile
  echo "Updated Dockerfile"
fi

############################
# 4. CHANGELOG
############################
if [[ -f docs/source/changelogs/changelog.rst ]]; then

    awk -v v="$VERSION" -v d="$(date +%Y/%m/%d)" '
    NR==2 {
    print
    print ""
    print "[" v "] - " d
    print "--------------------"
    print ""
    print "Added"
    print "^^^^^"
    print ""
    print "* TODO: describe additions"
    print ""
    print "Changed"
    print "^^^^^^^"
    print ""
    print "* TODO: describe changes"
    print ""
    print "Fixed"
    print "^^^^^"
    print ""
    print "* TODO: describe fixes"
    next
    }
    { print }
    ' docs/source/changelogs/changelog.rst > docs/source/changelogs/changelog.rst.new

    mv docs/source/changelogs/changelog.rst.new docs/source/changelogs/changelog.rst

    echo "Updated CHANGELOG"
fi


############################
# cleanup
############################
rm -f **.bak 2>/dev/null || true

echo "Release sync complete: $VERSION"
