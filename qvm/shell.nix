with import <nixpkgs> { };
let
  pythonEnv = python310.withPackages (ps: [
      ps.pip
      ps.numpy
      ps.pandas
      ps.matplotlib
    ]);
in
mkShell {
  buildInputs = [
    git
    gcc
    zlib
    pdm
    texlive.combined.scheme-full
  ];
  
  shellHook = ''
    export PATH=${pythonEnv}/bin:$PATH
    export PYTHONPATH=${gcc}/lib:${zlib}/lib:$PYTHONPATH
    export PYTHONPATH=$PYTHONPATH:$(pwd)
    export LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH
  ''; 
}
