{
  inputs,
  lib,
  config,
  pkgs,
  ...
}:

{
  perSystem =
    { system, pkgs, ... }:
    {
      packages.set-git-env = pkgs.writeShellApplication {
        name = "set-git-env";
        runtimeInputs = with pkgs; [
          git
          coreutils
          gnused
          gnugrep
        ];
        text = ''
          #!/usr/bin/env bash

          set -euo pipefail

          update_or_append() {
              local var_name=$1
              local var_value=$2
              local file=".env"

              # Create .env file if it doesn't exist
              if [ ! -f "$file" ]; then
                  touch "$file"
              fi
              
              if [ -s "$file" ] && [ "$(tail -c1 "$file"; echo x)" != $'\nx' ]; then
                  echo >> "$file"
              fi

              if grep -q "^$var_name=" "$file"; then
                  # Use sed with backup extension for compatibility
                  sed -i.bak "s/^$var_name=.*/$var_name=$var_value/" "$file"
                  # Remove backup file
                  rm -f "$file.bak"
              else
                  echo "$var_name=$var_value" >> "$file"
              fi
          }

          GIT_REPO_NAME=$(basename -s .git "$(git config --get remote.origin.url 2>/dev/null || echo "unknown-repo")")
          update_or_append "GIT_REPO_NAME" "$GIT_REPO_NAME"

          GIT_REF=$(git symbolic-ref -q --short HEAD 2>/dev/null || git rev-parse HEAD 2>/dev/null || echo "unknown-ref")
          update_or_append "GIT_REF" "$GIT_REF"

          GIT_SHA=$(git rev-parse HEAD 2>/dev/null || echo "unknown-sha")
          update_or_append "GIT_SHA" "$GIT_SHA"

          GIT_SHA_SHORT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
          update_or_append "GIT_SHA_SHORT" "$GIT_SHA_SHORT"
        '';
      };
    };
}