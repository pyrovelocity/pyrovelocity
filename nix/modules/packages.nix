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
      # Extract pyrovelocity package from each python set - following stormi simple pattern
      pyrovelocityPackages = lib.mapAttrs (_: pythonSet: pythonSet.pyrovelocity) pythonSets;
      editablePyrovelocityPackages = lib.mapAttrs (_: pythonSet: pythonSet.pyrovelocity) editablePythonSets;

    in
    {
      packages = rec {
        # Basic package exposures following stormi pattern
        py311 = pyrovelocityPackages.py311;
        py312 = pyrovelocityPackages.py312;
        
        default = py311;
      };
    };
}