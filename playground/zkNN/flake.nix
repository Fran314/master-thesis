{
  description = "zkNN: ZKP framework for neural networks";

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

        emp-tool = pkgs.stdenv.mkDerivation {
          pname = "emp-tool";
          version = "unstable-2025-02-17";
          src = pkgs.fetchFromGitHub {
            owner = "emp-toolkit";
            repo = "emp-tool";
            rev = "802b5d4fb7cc7fcaadd411cd6aa5e72ed4dd57fd";
            sha256 = "sha256-amWh3EyJiJyeNQph37GKk35scJZ4d7ddYfYlmOTBXlM=";
          };
          nativeBuildInputs = [ pkgs.cmake pkgs.pkg-config ];
          buildInputs = [ pkgs.openssl pkgs.boost pkgs.gmp ];
          # Nix strips -march=native; explicitly enable required x86 ISA extensions
          NIX_CFLAGS_COMPILE = "-maes -mssse3 -mrdseed -mpclmul -msse4.1 -msse4.2 -mavx";
          cmakeFlags = [
            "-DCMAKE_INSTALL_PREFIX=${placeholder "out"}"
          ];
        };

        emp-ot = pkgs.stdenv.mkDerivation {
          pname = "emp-ot";
          version = "unstable-2025-02-17";
          src = pkgs.fetchFromGitHub {
            owner = "emp-toolkit";
            repo = "emp-ot";
            rev = "a603ca0c77fcda37b9d088bd692111f67a4bef96";
            sha256 = "sha256-q/J/bRNBJIJsXW1PpVYeZCN7BzwoNwkNPdhiQubmyVo=";
          };
          nativeBuildInputs = [ pkgs.cmake pkgs.pkg-config ];
          buildInputs = [ emp-tool pkgs.openssl pkgs.boost pkgs.gmp ];
          inherit (emp-tool) NIX_CFLAGS_COMPILE;
          cmakeFlags = [
            "-DCMAKE_PREFIX_PATH=${emp-tool}"
          ];
        };

        emp-zk = pkgs.stdenv.mkDerivation {
          pname = "emp-zk";
          version = "unstable-2025-02-17";
          src = pkgs.fetchFromGitHub {
            owner = "emp-toolkit";
            repo = "emp-zk";
            rev = "73ab193a923d4c122be5a4f6bc1fe4f617966b02";
            sha256 = "sha256-o5cDmYajX3gjykGtfJX5ToSibHnBM+exf8v54Z5dJxg=";
          };
          nativeBuildInputs = [ pkgs.cmake pkgs.pkg-config ];
          buildInputs = [ emp-ot emp-tool pkgs.openssl pkgs.boost pkgs.gmp ];
          inherit (emp-tool) NIX_CFLAGS_COMPILE;
          cmakeFlags = [
            "-DCMAKE_PREFIX_PATH=${emp-ot};${emp-tool}"
          ];
        };

        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          torch
          torchvision
          numpy
        ]);

      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            #--- C++ BUILD DEPENDENCIES ---#
            pkgs.cmake
            pkgs.pkg-config
            pkgs.gmp
            pkgs.openssl
            pkgs.boost
            pkgs.libsodium

            #--- EMP TOOLKIT ---#
            emp-tool
            emp-ot
            emp-zk

            #--- PYTHON (model training/dumping) ---#
            pythonEnv
          ];
        };
      }
    );
}
