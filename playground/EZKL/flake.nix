{
  description = "EZKL playground setup";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    {
      self,
      nixpkgs,
      ...
    }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
      };

      version = "23.0.3";

      ezkl = pkgs.stdenv.mkDerivation {
        pname = "ezkl";
        inherit version;

        src = pkgs.fetchurl {
          url = "https://github.com/zkonduit/ezkl/releases/download/v${version}/build-artifacts.ezkl-linux-gnu.tar.gz";
          hash = "sha256-+J/Ps4ChncEppYBrhiMvqogrGjESRnTK6a/p/+UyfEQ=";
        };

        sourceRoot = ".";

        nativeBuildInputs = [ pkgs.autoPatchelfHook ];
        buildInputs = [
          pkgs.stdenv.cc.cc.lib # libstdc++, libgcc_s
        ];

        installPhase = ''
          mkdir -p $out/bin
          cp ezkl $out/bin/ezkl
        '';

        dontBuild = true;
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = [
          ezkl

          (pkgs.python3.withPackages (
            python-pkgs: with python-pkgs; [
              torch
              torchvision
              onnx
              onnxscript
            ]
          ))

          # only for the test script
          jq
        ];
      };
    };
}
