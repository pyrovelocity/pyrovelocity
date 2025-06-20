#!/usr/bin/env bash

set -euo pipefail

display_help() {
    cat << EOF
Usage: $0 [options]

create_worktree.sh creates a new git worktree for experimental PyroVelocity development.
The worktree is created in the workspace root directory next to the main repository,
ensuring uniform access to dependency code across all worktrees.

Options:
  -h, --help                  Display this help message and exit
  -w, --workspace-root PATH   Set the workspace root directory (required)
  -r, --repo-name NAME        Set the main repository name (default: pyrovelocity-01032024)
  -b, --branch-name NAME      Set the branch name for the worktree (required)
  -n, --worktree-name NAME    Set the worktree directory name (default: pyrovelocity-<branch>)
  -s, --spec                  Copy spec file to worktree (default: false)
  -v, --verbose               Enable verbose output
  -d, --dry-run               Perform a dry run without creating the worktree

Required Parameters:
  --workspace-root            The root workspace directory containing repositories
  --branch-name               The git branch name for the worktree

Examples:
  # Basic usage
  $0 --workspace-root ~/projects/pyrovelocity-workspace \\
     --branch-name feature/724-wtts

  # Custom worktree name with spec
  $0 --workspace-root ~/projects/pyrovelocity-workspace \\
     --branch-name feature/multiple-inference \\
     --worktree-name pyrovelocity-inference-experiment \\
     --spec

  # Dry run to see what would be created
  $0 --workspace-root ~/projects/pyrovelocity-workspace \\
     --branch-name feature/724-wtts \\
     --dry-run

EOF
}

# Default values
WORKSPACE_ROOT=""
REPO_NAME="pyrovelocity-01032024"
BRANCH_NAME=""
WORKTREE_NAME=""
COPY_SPEC=false
VERBOSE=false
DRY_RUN=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) display_help; exit 0 ;;
        -w|--workspace-root) WORKSPACE_ROOT="$2"; shift ;;
        -r|--repo-name) REPO_NAME="$2"; shift ;;
        -b|--branch-name) BRANCH_NAME="$2"; shift ;;
        -n|--worktree-name) WORKTREE_NAME="$2"; shift ;;
        -s|--spec) COPY_SPEC=true ;;
        -v|--verbose) VERBOSE=true ;;
        -d|--dry-run) DRY_RUN=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Validation functions
validate_required_parameters() {
    local errors=()
    
    if [ -z "$WORKSPACE_ROOT" ]; then
        errors+=("--workspace-root is required")
    fi
    
    if [ -z "$BRANCH_NAME" ]; then
        errors+=("--branch-name is required")
    fi
    
    if [ ${#errors[@]} -gt 0 ]; then
        echo "Error: Missing required parameters:"
        printf '  %s\n' "${errors[@]}"
        echo ""
        echo "Use --help for usage information."
        exit 1
    fi
}

validate_workspace_structure() {
    local workspace_root="$1"
    local repo_name="$2"
    local main_repo_path="${workspace_root}/${repo_name}"
    
    if [ ! -d "$workspace_root" ]; then
        echo "Error: Workspace root directory does not exist: $workspace_root"
        exit 1
    fi
    
    if [ ! -d "$main_repo_path" ]; then
        echo "Error: Main repository not found: $main_repo_path"
        exit 1
    fi
    
    if [ ! -d "${main_repo_path}/.git" ]; then
        echo "Error: Main repository is not a git repository: $main_repo_path"
        exit 1
    fi
}

validate_branch_name() {
    local branch_name="$1"
    
    # Check for valid git branch name characters
    if [[ ! "$branch_name" =~ ^[a-zA-Z0-9._/-]+$ ]]; then
        echo "Error: Invalid branch name '$branch_name'. Use alphanumeric characters, dots, underscores, hyphens, and forward slashes only."
        exit 1
    fi
    
    # Check length (Git has a 250 character limit for ref names)
    if [ ${#branch_name} -gt 200 ]; then
        echo "Error: Branch name too long (${#branch_name} characters). Keep it under 200 characters."
        exit 1
    fi
}

generate_worktree_name() {
    local branch_name="$1"
    
    # Convert branch name to worktree name if not specified
    if [ -z "$WORKTREE_NAME" ]; then
        # Remove 'feature/' prefix if present and sanitize
        local sanitized_branch="${branch_name#feature/}"
        sanitized_branch="${sanitized_branch//\//-}"  # Replace / with -
        WORKTREE_NAME="pyrovelocity-${sanitized_branch}"
    fi
}

check_worktree_conflicts() {
    local workspace_root="$1"
    local worktree_name="$2"
    local repo_name="$3"
    local worktree_path="${workspace_root}/${worktree_name}"
    local main_repo_path="${workspace_root}/${repo_name}"
    
    if [ -d "$worktree_path" ]; then
        echo "Error: Worktree directory already exists: $worktree_path"
        exit 1
    fi
    
    # Check if branch already exists as a worktree
    if git -C "$main_repo_path" worktree list | grep -q "\\[$BRANCH_NAME\\]"; then
        echo "Error: Branch '$BRANCH_NAME' is already checked out in another worktree:"
        git -C "$main_repo_path" worktree list | grep "\\[$BRANCH_NAME\\]"
        exit 1
    fi
}

copy_essential_files() {
    local main_repo_path="$1"
    local worktree_path="$2"
    local copy_spec="$3"
    local branch_name="$4"
    
    # List of essential untracked files to copy
    local essential_files=(
        "CLAUDE.md"
        ".env"
        ".envrc"
        "activate_env.sh"
        ".gitignore"
    )
    
    # Copy essential files if they exist (but skip CLAUDE.md since we'll create a custom one)
    for file in "${essential_files[@]}"; do
        if [ "$file" != "CLAUDE.md" ] && [ -f "${main_repo_path}/${file}" ]; then
            cp "${main_repo_path}/${file}" "${worktree_path}/"
            [ "$VERBOSE" = true ] && echo "Copied: $file"
        fi
    done
    
    # Copy spec file if requested
    if [ "$copy_spec" = true ]; then
        local spec_file="specs/${branch_name}-plan.md"
        if [ -f "${main_repo_path}/${spec_file}" ]; then
            # Create specs directory in worktree if it doesn't exist
            mkdir -p "${worktree_path}/specs"
            cp "${main_repo_path}/${spec_file}" "${worktree_path}/${spec_file}"
            [ "$VERBOSE" = true ] && echo "Copied: $spec_file"
        else
            echo "Warning: Spec file not found: $spec_file"
        fi
    fi
}

create_worktree_claude_config() {
    local worktree_path="$1"
    local workspace_root="$2"
    local worktree_name="$3"
    local branch_name="$4"
    local main_repo_path="$5"
    
    cat > "${worktree_path}/CLAUDE.md" << EOF
# CLAUDE.md - Experimental Worktree Configuration

## CRITICAL WORKING DIRECTORY CONTEXT
**You are working in a git worktree at:**
\`${workspace_root}/${worktree_name}\`

**NOT in the main repository at:**
\`${workspace_root}/$(basename "$main_repo_path")\`

## Worktree Information
- **Worktree Path**: \`${worktree_path}\`
- **Branch**: \`${branch_name}\`
- **Workspace Root**: \`${workspace_root}\`

## File Path Context
All relative paths are from the worktree root: \`${worktree_path}/\`
- Source code: \`src/pyrovelocity/models/modular/...\`
- Scripts: \`scripts/validation/...\`
- Tests: \`src/pyrovelocity/tests/...\`

## Workspace Structure
The workspace contains multiple repositories for dependency reference:
EOF

    # Add dependency repository information if available
    if [ -d "${workspace_root}/pyro" ]; then
        echo "- \`../pyro/\` - Pyro probabilistic programming framework source" >> "${worktree_path}/CLAUDE.md"
    fi
    if [ -d "${workspace_root}/scanpy" ]; then
        echo "- \`../scanpy/\` - Scanpy single-cell analysis toolkit source" >> "${worktree_path}/CLAUDE.md"
    fi
    if [ -d "${workspace_root}/anndata" ]; then
        echo "- \`../anndata/\` - AnnData annotated data structures source" >> "${worktree_path}/CLAUDE.md"
    fi
    
    # Include original CLAUDE.md content if it exists
    if [ -f "${main_repo_path}/CLAUDE.md" ]; then
        echo "" >> "${worktree_path}/CLAUDE.md"
        echo "## Original Project Configuration" >> "${worktree_path}/CLAUDE.md"
        echo "" >> "${worktree_path}/CLAUDE.md"
        cat "${main_repo_path}/CLAUDE.md" >> "${worktree_path}/CLAUDE.md"
    fi
}

create_worktree_readme() {
    local worktree_path="$1"
    local branch_name="$2"
    local creation_time="$3"
    
    cat > "${worktree_path}/WORKTREE_README.md" << EOF
# PyroVelocity Experimental Worktree

## Worktree Information
- **Created**: ${creation_time}
- **Branch**: \`${branch_name}\`
- **Purpose**: Experimental development isolated from main repository

## Quick Start
\`\`\`bash
# Activate virtual environment (if using one)
source .venv/bin/activate  # or source activate_env.sh

# Run tests
pytest src/pyrovelocity/tests/ -v

# Run validation scripts
python scripts/validation/posterior-predictive-check.py
\`\`\`

## Important Notes
- This is a git worktree sharing history with the main repository
- Changes here are isolated to the \`${branch_name}\` branch
- Use \`git worktree remove\` when finished with experiments
- Essential configuration files have been copied from main repository

## Cleanup
When finished with this worktree:
\`\`\`bash
# From the main repository
git worktree remove ${worktree_path}
git branch -d ${branch_name}  # if you want to delete the branch
\`\`\`
EOF
}

create_worktree() {
    local workspace_root="$1"
    local repo_name="$2"
    local branch_name="$3"
    local worktree_name="$4"
    local copy_spec="$5"
    
    local main_repo_path="${workspace_root}/${repo_name}"
    local worktree_path="${workspace_root}/${worktree_name}"
    local creation_time=$(date '+%Y-%m-%d %H:%M:%S')
    
    if [ "$DRY_RUN" = true ]; then
        echo "DRY RUN - Would create worktree with the following configuration:"
        echo "  Workspace Root: $workspace_root"
        echo "  Main Repository: $main_repo_path"
        echo "  Worktree Path: $worktree_path"
        echo "  Branch Name: $branch_name"
        echo "  Copy Spec: $copy_spec"
        echo ""
        echo "Files that would be copied:"
        for file in "CLAUDE.md" ".env" ".envrc" "activate_env.sh"; do
            if [ -f "${main_repo_path}/${file}" ]; then
                echo "  - $file"
            fi
        done
        if [ "$copy_spec" = true ]; then
            local spec_file="specs/${branch_name}-plan.md"
            if [ -f "${main_repo_path}/${spec_file}" ]; then
                echo "  - $spec_file"
            fi
        fi
        return 0
    fi
    
    [ "$VERBOSE" = true ] && set -x
    
    # Create the worktree (handle both new and existing branches)
    echo "Creating git worktree at: $worktree_path"
    
    # Check if branch already exists
    if git -C "$main_repo_path" show-ref --verify --quiet "refs/heads/$branch_name"; then
        echo "Branch '$branch_name' already exists, checking it out in worktree..."
        git -C "$main_repo_path" worktree add "$worktree_path" "$branch_name"
    else
        echo "Creating new branch '$branch_name' in worktree..."
        git -C "$main_repo_path" worktree add "$worktree_path" -b "$branch_name"
    fi
    
    # Copy essential files
    echo "Copying essential configuration files..."
    copy_essential_files "$main_repo_path" "$worktree_path" "$copy_spec" "$branch_name"
    
    # Create worktree-specific configuration
    if [ -f "${worktree_path}/CLAUDE.md" ]; then
        echo "Replacing git-copied CLAUDE.md with worktree-specific configuration..."
    else
        echo "Creating worktree-specific CLAUDE.md configuration..."
    fi
    create_worktree_claude_config "$worktree_path" "$workspace_root" "$worktree_name" "$branch_name" "$main_repo_path"
    create_worktree_readme "$worktree_path" "$branch_name" "$creation_time"
    
    [ "$VERBOSE" = true ] && set +x
    
    echo ""
    echo "✅ Worktree created successfully!"
    echo "📁 Location: $worktree_path"
    echo "🌿 Branch: $branch_name"
    echo ""
    echo "To start working:"
    echo "  cd $worktree_path"
    echo "  # CLAUDE.md contains worktree-specific configuration"
    echo "  # Review WORKTREE_README.md for usage instructions"
    echo ""
    echo "To remove when finished:"
    echo "  git -C $main_repo_path worktree remove $worktree_path"
}

main() {
    validate_required_parameters
    validate_workspace_structure "$WORKSPACE_ROOT" "$REPO_NAME"
    validate_branch_name "$BRANCH_NAME"
    generate_worktree_name "$BRANCH_NAME"
    check_worktree_conflicts "$WORKSPACE_ROOT" "$WORKTREE_NAME" "$REPO_NAME"
    
    if [ "$VERBOSE" = true ]; then
        echo "Configuration:"
        echo "  Workspace Root: $WORKSPACE_ROOT"
        echo "  Repository Name: $REPO_NAME" 
        echo "  Branch Name: $BRANCH_NAME"
        echo "  Worktree Name: $WORKTREE_NAME"
        echo "  Copy Spec: $COPY_SPEC"
        echo ""
    fi
    
    create_worktree "$WORKSPACE_ROOT" "$REPO_NAME" "$BRANCH_NAME" "$WORKTREE_NAME" "$COPY_SPEC"
}

# Check for required dependencies
type git &>/dev/null || { echo "git is required but not installed. Exiting."; exit 1; }

main