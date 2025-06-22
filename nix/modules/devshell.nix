{
  inputs,
  ...
}:
{
  perSystem =
    {
      config,
      self',
      pkgs,
      lib,
      baseWorkspace,
      editablePythonSets,
      pythonVersions,
      ...
    }:
    let
      mkDevShell =
        {
          name,
          pythonVersion,
        }:
        let
          python = pythonVersions.${pythonVersion};
          pythonEnv = editablePythonSets.${pythonVersion};
          virtualenv = pythonEnv.mkVirtualEnv "${name}-dev-env" baseWorkspace.deps.all;
        in
        pkgs.mkShell {
          inherit name;
          inputsFrom = [ config.pre-commit.devShell ];
          packages = with pkgs; [
            just
            uv
            virtualenv
            config.packages.set-git-env
          ];

          env = {
            UV_NO_SYNC = "1";
            UV_PYTHON = "${python}/bin/python";
            UV_PYTHON_DOWNLOADS = "never";
            QUARTO_PYTHON = "${python}/bin/python";
          };

          shellHook = ''
            unset PYTHONPATH
            export REPO_ROOT=$(git rev-parse --show-toplevel)
            set-git-env
          '';
        };
    in
    {
      devShells = rec {
        py311 = mkDevShell {
          name = "pyrovelocity-3.11";
          pythonVersion = "py311";
        };

        py312 = mkDevShell {
          name = "pyrovelocity-3.12";
          pythonVersion = "py312";
        };

        default = py311;
      };
    };
}