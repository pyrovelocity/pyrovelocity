{
  inputs,
  ...
}:
{
  perSystem =
    {
      config,
      self',
      inputs',
      pkgs,
      lib,
      system,
      pythonSets,
      editablePythonSets,
      pythonVersions,
      defaultPython,
      ...
    }:
    let
      # Extract pyrovelocity package from each python set
      pyrovelocityPackages = lib.mapAttrs (_: pythonSet: pythonSet.pyrovelocity) pythonSets;
      editablePyrovelocityPackages = lib.mapAttrs (_: pythonSet: pythonSet.pyrovelocity) editablePythonSets;

      # Create development environments
      mkDevEnv = name: pythonSet:
        pythonSet.python.withPackages (ps: [
          pythonSet.pyrovelocity
          # Add all dev dependencies
          ps.pytest
          ps.pytest-cov
          ps.pytest-xdist
          ps.pytest-asyncio
          ps.pytest-mock
          ps.pytest-bdd
          ps.black
          ps.ruff
          ps.mypy
          ps.pyright
          ps.pre-commit
          ps.pre-commit-hooks
          ps.ipdb
          ps.ipython
          ps.jupyter
          ps.jupyterlab
          ps.sphinx
          ps.furo
          ps.myst-parser
          ps.nbsphinx
        ]);

      # Create editable development environments
      mkEditableDevEnv = name: pythonSet:
        pythonSet.python.withPackages (ps: [
          pythonSet.pyrovelocity
          # Add all dev dependencies
          ps.pytest
          ps.pytest-cov
          ps.pytest-xdist
          ps.pytest-asyncio
          ps.pytest-mock
          ps.pytest-bdd
          ps.black
          ps.ruff
          ps.mypy
          ps.pyright
          ps.pre-commit
          ps.pre-commit-hooks
          ps.ipdb
          ps.ipython
          ps.jupyter
          ps.jupyterlab
          ps.sphinx
          ps.furo
          ps.myst-parser
          ps.nbsphinx
        ]);

      # Development environments
      devEnvironments = lib.mapAttrs mkDevEnv pythonSets;
      editableDevEnvironments = lib.mapAttrs mkEditableDevEnv editablePythonSets;

      # Default system packages for shells
      defaultSystemPackages = with pkgs; [
        git
        curl
        which
        gnumake
        openssl
        pkg-config
        stdenv.cc
        zlib
        libxml2
        libxslt
        readline
        sqlite
        tk
        zstd
        xz
        bzip2
        ncurses
        libffi
        expat
        mpdecimal
        libxcrypt
      ];

      # Extra system packages for CUDA support
      cudaSystemPackages = lib.optionals (pkgs.stdenv.isLinux) (with pkgs; [
        cudaPackages_12_1.cudatoolkit
        cudaPackages_12_1.cudnn
        linuxPackages.nvidia_x11
      ]);

      # Create development shells
      mkDevShell = name: env:
        pkgs.mkShell {
          name = "pyrovelocity-${name}-devshell";
          nativeBuildInputs = defaultSystemPackages ++ cudaSystemPackages;
          buildInputs = [ env ];
          shellHook = ''
            export PYTHONPATH="${env}/${env.python.sitePackages}:$PYTHONPATH"
            export LD_LIBRARY_PATH="${env}/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH"
            export QUARTO_PYTHON="${env}/bin/python"
            export REPO_ROOT="$(pwd)"
            
            echo "PyroVelocity development environment (${name})"
            echo "Python: $(${env}/bin/python --version)"
            echo "PyroVelocity: $(${env}/bin/python -c 'import pyrovelocity; print(pyrovelocity.__version__)' 2>/dev/null || echo 'not found')"
          '';
        };

      # Development shells
      devShells = lib.mapAttrs mkDevShell devEnvironments;
      editableDevShells = lib.mapAttrs (name: env: 
        mkDevShell "${name}-editable" env
      ) editableDevEnvironments;

      # Create release environment  
      releaseEnv = pkgs.buildEnv {
        name = "pyrovelocity-release-env";
        paths = with pkgs; [ git python311 ];
      };
    in
    {
      packages = 
        pyrovelocityPackages //
        editablePyrovelocityPackages //
        {
          inherit releaseEnv;
          default = pyrovelocityPackages.py311;
          
          # Expose development environments as packages
          pyrovelocityDevEnv310 = devEnvironments.py310;
          pyrovelocityDevEnv311 = devEnvironments.py311;
          pyrovelocityDevEnv312 = devEnvironments.py312;
          
          # Expose editable development environments
          pyrovelocityEditableDevEnv310 = editableDevEnvironments.py310;
          pyrovelocityEditableDevEnv311 = editableDevEnvironments.py311;
          pyrovelocityEditableDevEnv312 = editableDevEnvironments.py312;
        };

      devShells = 
        devShells //
        editableDevShells //
        {
          default = devShells.py311;
        };

      # Expose package sets for other modules
      _module.args = {
        inherit 
          pyrovelocityPackages
          editablePyrovelocityPackages
          devEnvironments
          editableDevEnvironments;
      };
    };
}