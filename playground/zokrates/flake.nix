{
  description = "ZoKrates playground setup";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      rust-overlay,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };

        zokrates = pkgs.rustPlatform.buildRustPackage rec {
          pname = "zokrates";
          version = "0.8.8";

          src = pkgs.fetchFromGitHub {
            owner = "Zokrates";
            repo = "ZoKrates";
            rev = version;
            hash = "sha256-EAxQPzAyJG4ybRHCUEwWtnlkpMhZ9YBqFtTZ6jOybko=";
          };

          cargoLock = {
            lockFile = "${src}/Cargo.lock";
            outputHashes = {
              # Git dependencies require output hashes
              "ark-marlin-0.3.0" = "sha256-dc4StG8FEDrxVuo00M/uF6SRi5rpTx4I2PnmKtVJTLI=";
              "phase2-0.2.2" = "sha256-eONGJEK6g2DN6dKL86vMVx/Md63u5E2Qzv4tpek0NzM=";
            };
          };

          # Only build the zokrates_cli package
          cargoBuildFlags = [
            "-p"
            "zokrates_cli"
          ];
          cargoTestFlags = [
            "-p"
            "zokrates_cli"
          ];

          # Tests require network access and other resources
          doCheck = false;

          # Install the stdlib alongside the binary
          postInstall = ''
            mkdir -p $out/share/zokrates
            cp -r ${src}/zokrates_stdlib/stdlib $out/share/zokrates/stdlib
          '';

          # Set up a wrapper script that sets ZOKRATES_STDLIB
          postFixup = ''
            wrapProgram $out/bin/zokrates \
              --set ZOKRATES_STDLIB $out/share/zokrates/stdlib
          '';

          nativeBuildInputs = with pkgs; [
            makeWrapper
          ];
        };

        lib = pkgs.lib;
      in
      {
        packages = {
          inherit zokrates;
          default = zokrates;
        };

        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            rust-bin.stable.latest.default
            zokrates
          ];
        };
      }
    );
}
