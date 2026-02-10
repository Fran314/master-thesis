{
  description = "Noir playground setup";

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

        rustToolchain = pkgs.rust-bin.stable.latest.default;

        # Use the overlay's toolchain so we satisfy Noir's MSRV (1.89.0)
        rustPlatform = pkgs.makeRustPlatform {
          cargo = rustToolchain;
          rustc = rustToolchain;
        };

        nargo = rustPlatform.buildRustPackage rec {
          pname = "nargo";
          version = "1.0.0-beta.18";

          src = pkgs.fetchFromGitHub {
            owner = "noir-lang";
            repo = "noir";
            rev = "v${version}";
            hash = "sha256-euY3XJ2jJEnb2XFY7xupKR3OtOepRqIj67X5TzRaJ9k=";
          };

          cargoLock = {
            lockFile = "${src}/Cargo.lock";
            # Automatically fetch any git dependencies referenced in Cargo.lock
            allowBuiltinFetchGit = true;
          };

          # Match noirup: RUSTFLAGS="-C target-cpu=native" cargo build --release
          RUSTFLAGS = "-C target-cpu=native";

          # The noirc_driver build script calls `git rev-parse HEAD`;
          # in the Nix sandbox there is no .git dir, so provide the value directly.
          GIT_COMMIT = "v${version}";
          GIT_DIRTY = "false";

          nativeBuildInputs = [ pkgs.git ];

          # Tests require network access and other resources
          doCheck = false;
        };

        lib = pkgs.lib;
      in
      {
        packages = {
          inherit nargo;
          default = nargo;
        };

        devShells.default = pkgs.mkShell {
          buildInputs = [
            rustToolchain
            nargo
          ];

          NARGO_HOME = builtins.toString ./. + "/.nargo";

          shellHook = ''
            export NARGO_HOME="$NARGO_HOME"
          '';
        };
      }
    );
}
