{
  description = "Flake setup to work with snarkjs + circom";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            #--- SNARKJS ---#
            (pkgs.callPackage ./snarkjs.nix { })
            #--- ---#

            #--- CIRCOM ---#
            circom

            # Circom dependencies for computing witness in C++
            nlohmann_json
            gmp
            nasm
            #--- ---#
          ];
        };
      }
    );
}
