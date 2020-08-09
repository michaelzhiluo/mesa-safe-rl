#!/usr/bin/env bash
# YAPF + Clang formatter (if installed). This script formats all changed files from the last mergebase.
# You are encouraged to run this locally before pushing changes for review.

# Cause the script to exit if a single command fails
set -euo pipefail

FLAKE8_VERSION_REQUIRED="3.7.7"
YAPF_VERSION_REQUIRED="0.23.0"

check_command_exist() {
    VERSION=""
    case "$1" in
        yapf)
            VERSION=$YAPF_VERSION_REQUIRED
            ;;
        flake8)
            VERSION=$FLAKE8_VERSION_REQUIRED
            ;;
        *)
            echo "$1 is not a required dependency"
            exit 1
    esac
    if ! [ -x "$(command -v "$1")" ]; then
        echo "$1 not installed. pip install $1==$VERSION"
        exit 1
    fi
}

check_command_exist yapf
check_command_exist flake8

ver=$(yapf --version)
if ! echo "$ver" | grep -q 0.23.0; then
    echo "Wrong YAPF version installed: 0.23.0 is required, not $ver. $YAPF_DOWNLOAD_COMMAND_MSG"
    exit 1
fi

# this stops git rev-parse from failing if we run this from the .git directory
builtin cd "$(dirname "${BASH_SOURCE:-$0}")"

ROOT="$(git rev-parse --show-toplevel)"
builtin cd "$ROOT" || exit 1

FLAKE8_VERSION=$(flake8 --version | awk '{print $1}')
YAPF_VERSION=$(yapf --version | awk '{print $2}')

# params: tool name, tool version, required version
tool_version_check() {
    if [ "$2" != "$3" ]; then
        echo "WARNING: Ray uses $1 $3, You currently are using $2. This might generate different results."
    fi
}

tool_version_check "flake8" "$FLAKE8_VERSION" "$FLAKE8_VERSION_REQUIRED"
tool_version_check "yapf" "$YAPF_VERSION" "$YAPF_VERSION_REQUIRED"

YAPF_FLAGS=(
    '--recursive'
    '--parallel'
)

YAPF_EXCLUDES=(
    #'--exclude' 'env/'
)

GIT_LS_EXCLUDES=(
)


FLAKE8_EXCLUDE=""
FLAKE8_IGNORES="--ignore=C408,E121,E123,E126,E226,E24,E704,W503,W504,W605"
FLAKE8_PYX_IGNORES="--ignore=C408,E121,E123,E126,E211,E225,E226,E227,E24,E704,E999,W503,W504,W605"


# Format all files, and print the diff to stdout for travis.
format_all() {
    command -v flake8 &> /dev/null;
    HAS_FLAKE8=$?

    echo "$(date)" "YAPF...."
    git ls-files -- '*.py' "${GIT_LS_EXCLUDES[@]}" | xargs -P 10 \
      yapf --in-place "${YAPF_EXCLUDES[@]}" "${YAPF_FLAGS[@]}"
    if [ $HAS_FLAKE8 ]; then
      echo "$(date)" "Flake8...."
      git ls-files -- '*.py' "${GIT_LS_EXCLUDES[@]}" | xargs -P 5 \
        flake8 --inline-quotes '"' --no-avoid-escape  "$FLAKE8_EXCLUDE" "$FLAKE8_IGNORES"

      git ls-files -- '*.pyx' '*.pxd' '*.pxi' "${GIT_LS_EXCLUDES[@]}" | xargs -P 5 \
        flake8 --inline-quotes '"' --no-avoid-escape "$FLAKE8_EXCLUDE" "$FLAKE8_PYX_IGNORES"
    fi

    echo "$(date)" "clang-format...."
    if command -v clang-format >/dev/null; then
      git ls-files -- '*.cc' '*.h' "${GIT_LS_EXCLUDES[@]}" | xargs -P 5 clang-format -i
    fi

    echo "$(date)" "done!"
}

format_all "${@}"
if [ -n "${FORMAT_SH_PRINT_DIFF-}" ]; then git --no-pager diff; fi

PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-python}


if ! git diff --quiet &>/dev/null; then
    echo 'Reformatted changed files. Please review and stage the changes.'
    echo 'Files updated:'
    echo

    git --no-pager diff --name-only

    exit 1
fi