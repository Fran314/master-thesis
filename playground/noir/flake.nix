{
  description = "Noir playground setup";

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

      lib = pkgs.lib;

      version = "1.0.0-beta.18";

      # --- Precompiled tarball ---
      nargo-tarball = pkgs.stdenv.mkDerivation {
        pname = "nargo";
        inherit version;

        src = pkgs.fetchurl {
          url = "https://github.com/noir-lang/noir/releases/download/v${version}/noir-x86_64-unknown-linux-gnu.tar.gz";
          hash = "sha256-J6el8m5OKp+TC02EphaYHVm11IEQWP3kjfJfrLW3Tr4=";
        };

        sourceRoot = ".";

        # autoPatchelfHook rewrites the ELF interpreter and RPATH so
        # these precompiled binaries find glibc & friends in the Nix store.
        nativeBuildInputs = [ pkgs.autoPatchelfHook ];

        # Libraries the binaries are linked against (use `ldd` to check)
        buildInputs = [
          pkgs.stdenv.cc.cc.lib # libstdc++, libgcc_s
          pkgs.zlib
        ];

        installPhase = ''
          mkdir -p $out/bin
          for bin in nargo noir-profiler noir-inspector; do
            [ -f "$bin" ] && install -m755 "$bin" "$out/bin/"
          done
        '';

        # No build step needed
        dontBuild = true;
      };

      # --- Build from source ---
      nargo-from-source = pkgs.rustPlatform.buildRustPackage rec {
        pname = "nargo";
        inherit version;

        src = pkgs.fetchFromGitHub {
          owner = "noir-lang";
          repo = "noir";
          rev = "v${version}";
          hash = "sha256-euY3XJ2jJEnb2XFY7xupKR3OtOepRqIj67X5TzRaJ9k=";
        };

        cargoLock = {
          lockFile = "${src}/Cargo.lock";
          allowBuiltinFetchGit = true;
        };

        # Same flags as from the `noirup` script
        RUSTFLAGS = "-C target-cpu=native";

        # Needed because `noirc_driver` build script requires being in a git repo(?)
        GIT_COMMIT = "v${version}";
        GIT_DIRTY = "false";

        doCheck = false;
      };

      # --- Switch between the two by changing this line ---
      nargo = nargo-tarball;
      # nargo = nargo-from-source;
    in
    {
      packages.${system} = {
        inherit nargo nargo-tarball nargo-from-source;
        default = nargo;
      };

      devShells.${system}.default = pkgs.mkShell {
        buildInputs = [
          nargo
        ];

        # Not sure if this variable is actually needed by `nargo` but doesn't hurt to add it
        NARGO_HOME = builtins.toString ./. + "/.nargo";

        shellHook = ''
          export NARGO_HOME="$NARGO_HOME"
        '';
      };
    };
}
