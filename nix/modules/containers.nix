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
      editableDevEnvironments,
      ...
    }:
    let
      isLinux = pkgs.stdenv.isLinux;
      
      # Configuration
      gitHubOrg = "pinellolab";
      packageName = "pyrovelocity";
      version = builtins.getEnv "VERSION";
      
      # Build configuration
      includedSystems = let
        envVar = builtins.getEnv "NIX_IMAGE_SYSTEMS";
      in
        if envVar == ""
        then ["x86_64-linux" "aarch64-linux"]
        else builtins.filter (sys: sys != "") (builtins.split " " envVar);
      
      # Default to py311 environment from uv2nix
      # Note: editableDevEnvironments already contains full python environments
      # with all dev dependencies included (see packages.nix mkEditableDevEnv)
      defaultPythonEnv = editableDevEnvironments.py311;
      
      # Container system packages (moved from legacy nix/pkgs)
      containerSysPackages = with pkgs; [
        bashInteractive
        cacert
        coreutils
        direnv
        gnutar
        gzip
        less
        nix
        procps
        sudo
        zsh
      ];

      containerDevPackages = with pkgs; [
        git
        helix
        lazygit
        neovim
        starship
        uv  # UV for dependency management
      ];
      
      # Container configuration
      buildMultiUserNixImage = import ("${inputs.nixpod.outPath}" + "/containers/nix.nix");
      
      # Import existing container definitions with uv2nix environments
      containerImage = import ../containers/default.nix {
        inherit pkgs gitHubOrg packageName;
        sysPackages = containerSysPackages;
        devPackages = containerDevPackages;
        extraSysPackages = [];  # Not used
        editableDevEnvironments = editableDevEnvironments;
      };
      
      codeImage = import ../containers/code.nix {
        inherit pkgs buildMultiUserNixImage;
        devPackages = containerDevPackages;
        sudoImage = inputs'.nixpod.packages.sudoImage;
        homeActivationPackage = inputs'.nixpod.legacyPackages.homeConfigurations.jovyan.activationPackage;
        pythonPackageEnv = defaultPythonEnv;
      };
      
      jupyterImage = import ../containers/jupyter.nix {
        inherit pkgs buildMultiUserNixImage;
        devPackages = containerDevPackages;
        sudoImage = inputs'.nixpod.packages.sudoImage;
        homeActivationPackage = inputs'.nixpod.legacyPackages.homeConfigurations.jovyan.activationPackage;
        pythonPackageEnv = defaultPythonEnv;
      };
    in
    {
      packages = lib.optionalAttrs isLinux {
        # Base containers
        baseContainerImage = containerImage.baseContainerImage;
        baseDevContainerImage = containerImage.baseDevContainerImage;
        
        # Main containers
        containerImage = containerImage.containerImage;
        devcontainerImage = containerImage.devcontainerImage;
        
        # Specialized containers
        pyrovelocityCodeImage = codeImage;
        pyrovelocityJupyterImage = jupyterImage;
      };
      
      legacyPackages = lib.optionalAttrs isLinux {
        # Docker manifest for multi-arch images
        pyrovelocityManifest = inputs.flocken.legacyPackages.${system}.mkDockerManifest {
          inherit version;
          github = {
            enable = true;
            enableRegistry = true;
            token = "$GH_TOKEN";
          };
          autoTags.branch = false;
          registries = {
            "ghcr.io" = {
              repo = lib.mkForce "${gitHubOrg}/${packageName}";
            };
          };
          imageFiles = map (sys: inputs.self.packages.${sys}.containerImage) includedSystems;
          tags = [
            (builtins.getEnv "GIT_SHA_SHORT")
            (builtins.getEnv "GIT_SHA")
            (builtins.getEnv "GIT_REF")
            "main"
          ];
        };
        
        pyrovelocity-devManifest = inputs.flocken.legacyPackages.${system}.mkDockerManifest {
          inherit version;
          github = {
            enable = true;
            enableRegistry = true;
            token = "$GH_TOKEN";
          };
          autoTags.branch = false;
          registries = {
            "ghcr.io" = {
              repo = lib.mkForce "${gitHubOrg}/${packageName}dev";
            };
          };
          imageFiles = map (sys: inputs.self.packages.${sys}.devcontainerImage) includedSystems;
          tags = [
            (builtins.getEnv "GIT_SHA_SHORT")
            (builtins.getEnv "GIT_SHA")
            (builtins.getEnv "GIT_REF")
            "dev"
          ];
        };
        
        pyrovelocity-codeManifest = inputs.flocken.legacyPackages.${system}.mkDockerManifest {
          inherit version;
          github = {
            enable = true;
            enableRegistry = true;
            token = "$GH_TOKEN";
          };
          autoTags.branch = false;
          registries = {
            "ghcr.io" = {
              repo = lib.mkForce "${gitHubOrg}/${packageName}code";
            };
          };
          imageFiles = map (sys: inputs.self.packages.${sys}.pyrovelocityCodeImage) includedSystems;
          tags = [
            (builtins.getEnv "GIT_SHA_SHORT")
            (builtins.getEnv "GIT_SHA")
            (builtins.getEnv "GIT_REF")
            "main"
          ];
        };
        
        pyrovelocity-jupyterManifest = inputs.flocken.legacyPackages.${system}.mkDockerManifest {
          inherit version;
          github = {
            enable = true;
            enableRegistry = true;
            token = "$GH_TOKEN";
          };
          autoTags.branch = false;
          registries = {
            "ghcr.io" = {
              repo = lib.mkForce "${gitHubOrg}/${packageName}jupyter";
            };
          };
          imageFiles = map (sys: inputs.self.packages.${sys}.pyrovelocityJupyterImage) includedSystems;
          tags = [
            (builtins.getEnv "GIT_SHA_SHORT")
            (builtins.getEnv "GIT_SHA")
            (builtins.getEnv "GIT_REF")
            "main"
          ];
        };
      };
    };
}