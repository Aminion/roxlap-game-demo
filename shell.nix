let
  rust_overlay = import (builtins.fetchTarball "https://github.com/oxalica/rust-overlay/archive/master.tar.gz");
  pkgs = import <nixpkgs> { overlays = [ rust_overlay ]; };
  rustToolchain = pkgs.rust-bin.nightly.latest.default.override {
    extensions = [ "rust-src" "rust-analyzer" ];
  };
in
pkgs.mkShell {
  buildInputs = with pkgs; [
    rustToolchain
    SDL2
    SDL2_mixer
    SDL2_gfx
    SDL2_ttf
    open-sans
    vulkan-loader
    vulkan-validation-layers
    wayland
    libxkbcommon
  ];

  shellHook = ''
    ln -sf ${pkgs.open-sans}/share/fonts/truetype ./
    export VK_ICD_FILENAMES=/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.json
    export LD_LIBRARY_PATH="${pkgs.vulkan-loader}/lib:${pkgs.wayland}/lib:${pkgs.libxkbcommon}/lib:$LD_LIBRARY_PATH"
  '';
}