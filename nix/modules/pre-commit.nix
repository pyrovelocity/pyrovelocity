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
      ...
    }:
    {
      # Provide a simple pre-commit devShell for compatibility
      # This creates an empty devShell that other modules can reference
      pre-commit.devShell = pkgs.mkShell {
        name = "pre-commit-shell";
        packages = with pkgs; [
          pre-commit
          ruff
          pyright
        ];
      };
    };
}