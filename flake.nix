{
  inputs = {
    # nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixpkgs.url = "github:NixOS/nixpkgs/24.11-beta";
    systems.url = "github:nix-systems/default";
    flake-parts.url = "github:hercules-ci/flake-parts";
    flake-utils.url = github:numtide/flake-utils;
    gitignore = {
      url = "github:hercules-ci/gitignore.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    
    # uv2nix inputs following python-nix-template pattern
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    
    # poetry2nix removed - migration to uv2nix complete
    
    flocken = {
      url = "github:mirkolenz/flocken/v2";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-parts.follows = "flake-parts";
      inputs.systems.follows = "systems";
    };
    home-manager = {
      url = "github:nix-community/home-manager/release-24.11";
      inputs = {
        nixpkgs.follows = "nixpkgs";
      };
    };
    nixpod = {
      url = "github:cameronraysmith/nixpod";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-parts.follows = "flake-parts";
      inputs.flocken.follows = "flocken";
      inputs.home-manager.follows = "home-manager";
      inputs.systems.follows = "systems";
    };
  };

  nixConfig = {
    extra-trusted-public-keys = [
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
      "pyproject-nix.cachix.org-1:UNzugsOlQIu2iOz0VyZNBQm2JSrL/kwxeCcFGw+jMe0="
      "pyrovelocity.cachix.org-1:+aX2YY45ZywieTsD2CnXLedN8RfKuRl6vL7+rLTCgnc="
    ];
    extra-substituters = [
      "https://nix-community.cachix.org"
      "https://pyproject-nix.cachix.org"
      "https://pyrovelocity.cachix.org"
    ];
    download-buffer-size = 524288000; # 500 MiB (1024 * 1024 * 500)
  };

  outputs = inputs @ {
    self,
    flake-parts,
    ...
  }:
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = import inputs.systems;
      
      # Import all modules from nix/modules automatically following python-nix-template pattern
      imports = with builtins; map (fn: ./nix/modules/${fn}) (attrNames (readDir ./nix/modules));

      perSystem = {
        self',
        inputs',
        pkgs,
        lib,
        system,
        config,
        ...
      }: let
        # Configuration
        gitHubOrg = "pinellolab";
        packageName = "pyrovelocity";
        version = builtins.getEnv "VERSION";
        isVersionNonEmpty = builtins.isString version && builtins.stringLength version > 0;
        gcpProjectId = builtins.getEnv "GCP_PROJECT_ID";

        # System configuration for container builds
        includedSystems = let
          envVar = builtins.getEnv "NIX_IMAGE_SYSTEMS";
        in
          if envVar == ""
          then ["x86_64-linux" "aarch64-linux"]
          else builtins.filter (sys: sys != "") (builtins.split " " envVar);

        # All functionality moved to nix/modules/ - no legacy imports needed
      in {
        formatter = pkgs.alejandra;

        # Note: devShells and packages are provided by modules/packages.nix
        
        # Configure nixpkgs with overlays - keeping minimal for uv2nix migration
        _module.args.pkgs = import inputs.nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
          };
          overlays = [
            inputs.gitignore.overlay
          ];
        };
      };
    };
}
