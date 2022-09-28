# Run with nixGL, eg `nixGLNvidia-510.47.03 python cifar10_convnet_run.py --test`

# To prevent JAX from allocating all GPU memory: XLA_PYTHON_CLIENT_PREALLOCATE=false
# To push build to cachix: nix-store -qR --include-outputs $(nix-instantiate shell.nix) | cachix push ploop

let
  # pkgs = import (/home/skainswo/dev/nixpkgs) { };

  # Last updated: 2022-05-16. Check for new commits at status.nixos.org.
  pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/556ce9a40abde33738e6c9eac65f965a8be3b623.tar.gz") {
    config.allowUnfree = true;
    # These actually cause problems for some reason. bug report?
    # config.cudaSupport = true;
    # config.cudnnSupport = true;
  };
in
pkgs.mkShell {
  buildInputs = with pkgs; [
    ffmpeg
    python3
    python3Packages.augmax
    python3Packages.einops
    python3Packages.flax
    python3Packages.ipython
    python3Packages.jax
    # See https://discourse.nixos.org/t/petition-to-build-and-cache-unfree-packages-on-cache-nixos-org/17440/14
    # as to why we don't use the source builds of jaxlib/tensorflow.
    (python3Packages.jaxlib-bin.override {
      cudaSupport = true;
    })
    python3Packages.matplotlib
    # python3Packages.pandas
    python3Packages.plotly
    # python3Packages.scikit-learn
    python3Packages.seaborn
    (python3Packages.tensorflow-bin.override {
      cudaSupport = false;
    })
    # Thankfully tensorflow-datasets does not have tensorflow as a propagatedBuildInput. If that were the case for any
    # of these dependencies, we'd be in trouble since Python does not like multiple versions of the same package in
    # PYTHONPATH.
    python3Packages.tensorflow-datasets
    python3Packages.tqdm
    python3Packages.wandb

    # Necessary for LaTeX in matplotlib.
    texlive.combined.scheme-full

    yapf
  ];

  # Don't clog EFS with wandb results. Wandb will create and use /tmp/wandb.
  WANDB_DIR = "/tmp";
  WANDB_CACHE_DIR = "/tmp";
}
