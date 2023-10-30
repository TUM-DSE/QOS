#{pkgs ? import <nixpkgs> {}}:
#
#pkgs.mkShell
#{
#    nativeBuildInputs = with pkgs;
#    [
#        
#    ];
#}

with import <nixpkgs> {};
mkShell rec {
  NIX_LD_LIBRARY_PATH = lib.makeLibraryPath [
    redis
    stdenv.cc.cc
    zlib
    pdm
  ];
  LD_LIBRARY_PATH = NIX_LD_LIBRARY_PATH;
  NIX_LD = lib.fileContents "${stdenv.cc}/nix-support/dynamic-linker";
  shellHook = ''
  source bin/activate
  '';
}
