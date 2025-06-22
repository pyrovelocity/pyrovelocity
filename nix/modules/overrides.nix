{
  inputs,
  ...
}:
{
  perSystem =
    {
      config,
      self',
      pkgs,
      lib,
      ...
    }:
    let
      mkPackageOverrides =
        { pkgs }:
        final: prev:
        let
          # Helper function for NVIDIA package postFixup
          nvidiaPostFixup = ''
            rm -r $out/${final.python.sitePackages}/nvidia/{__pycache__,__init__.py}
          '';

          # Helper function to apply postFixup to NVIDIA packages
          applyNvidiaPostFixup = name:
            prev.${name}.overrideAttrs (old: {
              postFixup = old.postFixup or "" + nvidiaPostFixup;
            });

          # NVIDIA CUDA packages that need postFixup only
          nvidiaCudaPostFixupOnlyPackages = lib.genAttrs [
            "nvidia-cublas-cu12"
            "nvidia-cufft-cu12"
            "nvidia-curand-cu12"
            "nvidia-cuda-cupti-cu12"
            "nvidia-cuda-nvrtc-cu12"
            "nvidia-cuda-runtime-cu12"
            "nvidia-nccl-cu12"
            "nvidia-nvtx-cu12"
          ] applyNvidiaPostFixup;

          # Build input overrides for packages requiring specific build dependencies
          buildInputsOverrides = {
            antlr4-python3-runtime = prev.antlr4-python3-runtime.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ final.setuptools ];
            });
            asciitree = prev.asciitree.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ final.setuptools ];
            });            
            cloudpickle = prev.cloudpickle.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ final.flit-core ];
            });
            docrep = prev.docrep.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ final.setuptools ];
            });            
            duckdb = prev.duckdb.overrideAttrs (old: {
              nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
                (final.resolveBuildSystem {
                  pybind11 = [ ];
                })
              ];
              buildInputs = (old.buildInputs or [ ]) ++ [ final.setuptools ];
            });            
            google-crc32c = prev.google-crc32c.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ final.setuptools ];
            });            
            feather-format = prev.feather-format.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ final.setuptools ];
            });
            hydra-core = prev.hydra-core.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ final.setuptools ];
            });
            hydra-joblib-launcher = prev.hydra-joblib-launcher.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ final.setuptools ];
            });
            hydra-zen = prev.hydra-zen.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ final.setuptools ];
            });
            loompy = prev.loompy.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ final.setuptools ];
            });            
            marshmallow-jsonschema = prev.marshmallow-jsonschema.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ final.setuptools ];
            });
            matplotlib-venn = prev.matplotlib-venn.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ final.setuptools ];
            });
            memoized-property = prev.memoized-property.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ final.setuptools ];
            });
            numba = prev.numba.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ pkgs.tbb_2021_11 ];
            });
            pqdata = prev.pqdata.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ 
                final.hatchling 
                final.hatch-vcs 
                final.pathspec 
                final.pluggy 
                final.setuptools-scm 
                final.setuptools 
                final.trove-classifiers
              ];
            });
            progressbar33 = prev.progressbar33.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ final.setuptools ];
            });            
            pyperclip = prev.pyperclip.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ final.setuptools ];
            });            
            ruamel-yaml-clib = prev.ruamel-yaml-clib.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ final.setuptools ];
            });
            scalene = prev.scalene.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ final.setuptools ];
            });
            scikit-learn = prev.scikit-learn.overrideAttrs (old: {
              nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
                (final.resolveBuildSystem {
                  cython = [ ];
                  meson = [ ];
                  meson-python = [ ];
                  ninja = [ ];
                  numpy = [ ];
                  scipy = [ ];
                })
              ];
            });
            scipy = prev.scipy.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [
                pkgs.openblas
                pkgs.pkg-config
                pkgs.suitesparse
              ];
              dontUseCmakeConfigure = true;
              nativeBuildInputs =
                (old.nativeBuildInputs or [ ])
                ++ [
                  pkgs.cmake
                  pkgs.gfortran
                  pkgs.meson
                  pkgs.pkg-config
                ]
                ++ [
                  (final.resolveBuildSystem {
                    cython = [ ];
                    meson-python = [ ];
                    ninja = [ ];
                    numpy = [ ];
                    pybind11 = [ ];
                    pythran = [ ];
                    wheel = [ ];
                  })
                ];
            });
            session-info = prev.session-info.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ final.setuptools ];
            });
            tinytimer = prev.tinytimer.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ final.setuptools ];
            });
            typechecks = prev.typechecks.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ final.setuptools ];
            });
            untokenize = prev.untokenize.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ final.setuptools ];
            });
            xdoctest = prev.xdoctest.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ final.setuptools ];
            });
          };

          # Platform-specific overrides
          conditionalOverrides =
            if pkgs.stdenv.isDarwin then
              {
                # grpcio = prev.grpcio.override { preferWheel = true; };
              }
            else if pkgs.stdenv.hostPlatform.system == "x86_64-linux" then
              {
                torch = prev.torch.overrideAttrs (attrs: {
                  nativeBuildInputs = attrs.nativeBuildInputs or [ ] ++ [ pkgs.autoPatchelfHook ];
                  buildInputs = attrs.buildInputs or [ ] ++ [
                    final.nvidia-cublas-cu12
                    final.nvidia-cuda-cupti-cu12
                    final.nvidia-cuda-nvrtc-cu12
                    final.nvidia-cuda-runtime-cu12
                    final.nvidia-cudnn-cu12
                    final.nvidia-cufft-cu12
                    final.nvidia-curand-cu12
                    final.nvidia-cusolver-cu12
                    final.nvidia-cusparse-cu12
                    final.nvidia-nccl-cu12
                    final.nvidia-nvjitlink-cu12
                    final.nvidia-nvtx-cu12
                  ];
                  postInstall = ''
                    addAutoPatchelfSearchPath "${final.nvidia-cublas-cu12}/${final.python.sitePackages}/nvidia/cublas/lib"
                    addAutoPatchelfSearchPath "${final.nvidia-cuda-cupti-cu12}/${final.python.sitePackages}/nvidia/cuda_cupti/lib"
                    addAutoPatchelfSearchPath "${final.nvidia-cuda-nvrtc-cu12}/${final.python.sitePackages}/nvidia/cuda_nvrtc/lib"
                    addAutoPatchelfSearchPath "${final.nvidia-cuda-runtime-cu12}/${final.python.sitePackages}/nvidia/cuda_runtime/lib"
                    addAutoPatchelfSearchPath "${final.nvidia-cudnn-cu12}/${final.python.sitePackages}/nvidia/cudnn/lib"
                    addAutoPatchelfSearchPath "${final.nvidia-cufft-cu12}/${final.python.sitePackages}/nvidia/cufft/lib"
                    addAutoPatchelfSearchPath "${final.nvidia-curand-cu12}/${final.python.sitePackages}/nvidia/curand/lib"
                    addAutoPatchelfSearchPath "${final.nvidia-cusolver-cu12}/${final.python.sitePackages}/nvidia/cusolver/lib"
                    addAutoPatchelfSearchPath "${final.nvidia-cusparse-cu12}/${final.python.sitePackages}/nvidia/cusparse/lib"
                    addAutoPatchelfSearchPath "${final.nvidia-nccl-cu12}/${final.python.sitePackages}/nvidia/nccl/lib"
                    addAutoPatchelfSearchPath "${final.nvidia-nvjitlink-cu12}/${final.python.sitePackages}/nvidia/nvjitlink/lib"
                    addAutoPatchelfSearchPath "${final.nvidia-nvtx-cu12}/${final.python.sitePackages}/nvidia/nvtx/lib"
                  '';
                  # Libraries that are expected to be provided by the NVIDIA driver at runtime
                  # and don't exist in the Nix store
                  autoPatchelfIgnoreMissingDeps = [
                    "libcuda.so.1"
                    "libcusparseLt.so.0"
                  ];
                });
              }
            else
              { };

          # Wheel preference overrides
          wheelOverrides = {
            # dm-tree = prev.dm-tree.override { preferWheel = true; };
            # duckdb = prev.duckdb.override { preferWheel = true; };
            # h5py = prev.h5py.override { preferWheel = true; };
            # hydra-core = prev.hydra-core.override { preferWheel = true; };
            # hydra-joblib-launcher = prev.hydra-joblib-launcher.override { preferWheel = true; };
            # mkdocs-material = prev.mkdocs-material.override { preferWheel = false; };
            # pyarrow = prev.pyarrow.override { preferWheel = true; };
            # scikit-learn = prev.scikit-learn.override { preferWheel = true; };
            # scipy = prev.scipy.override { preferWheel = true; };
            # tensorstore = prev.tensorstore.override { preferWheel = true; };
            # yarl = prev.yarl.override { preferWheel = true; };
          };

          # Special NVIDIA packages requiring complex overrides
          specialNvidiaOverrides = {
            nvidia-cudnn-cu12 = prev.nvidia-cudnn-cu12.overrideAttrs (old: {
              nativeBuildInputs = old.nativeBuildInputs or [ ] ++ [ pkgs.autoPatchelfHook ];
              propagatedBuildInputs =
                old.propagatedBuildInputs or [ ]
                ++ [
                  final.nvidia-cublas-cu12
                  pkgs.cudaPackages_12_1.cudnn
                ];
              postFixup = nvidiaPostFixup;
            });

            nvidia-cusparse-cu12 = prev.nvidia-cusparse-cu12.overrideAttrs (old: {
              propagatedBuildInputs =
                old.propagatedBuildInputs or [ ]
                ++ [
                  final.nvidia-nvjitlink-cu12
                  pkgs.cudaPackages_12_1.libnvjitlink
                ];
              postFixup = nvidiaPostFixup;
            });

            nvidia-cusolver-cu12 = prev.nvidia-cusolver-cu12.overrideAttrs (old: {
              propagatedBuildInputs =
                old.propagatedBuildInputs or [ ]
                ++ [
                  final.nvidia-cublas-cu12
                  final.nvidia-cusparse-cu12
                  pkgs.cudaPackages_12_1.libcublas
                  pkgs.cudaPackages_12_1.libcusparse
                ];
              postFixup = nvidiaPostFixup;
            });
          };

          # Package-specific overrides
          packageSpecificOverrides = {
            gtfparse = prev.gtfparse.overrideAttrs (_old: {
              postInstall = ''
                rm -f $out/lib/python3.10/site-packages/requirements.txt
                rm -f $out/lib/python3.11/site-packages/requirements.txt
                rm -f $out/lib/python3.12/site-packages/requirements.txt
              '';
            });

            optax = prev.optax.overrideAttrs (_old: {
              postInstall = ''
                rm -f $out/lib/python3.10/site-packages/docs/conf.py
                rm -fr $out/lib/python3.10/site-packages/docs/__pycache__
                rm -f $out/lib/python3.11/site-packages/docs/conf.py
                rm -fr $out/lib/python3.11/site-packages/docs/__pycache__
                rm -f $out/lib/python3.12/site-packages/docs/conf.py
                rm -fr $out/lib/python3.12/site-packages/docs/__pycache__
              '';
            });
          };
        in
        lib.foldl lib.mergeAttrs { } [
          buildInputsOverrides
          nvidiaCudaPostFixupOnlyPackages
          conditionalOverrides
          # wheelOverrides
          specialNvidiaOverrides
          packageSpecificOverrides
        ];

      mkSdistOverrides =
        { pkgs }:
        final: prev: {
          # Example overrides for source distributions
          # pyzmq = prev.pyzmq.overrideAttrs (old: {
          #   buildInputs = (old.buildInputs or []) ++ [ pkgs.zeromq ];
          #   nativeBuildInputs = (old.nativeBuildInputs or []) ++ [
          #     (final.resolveBuildSystem {
          #       cmake = [];
          #       ninja = [];
          #       packaging = [];
          #     })
          #   ];
          # });
        };
    in
    {
      _module.args = {
        packageOverrides = mkPackageOverrides { inherit pkgs; };
        sdistOverrides = mkSdistOverrides { inherit pkgs; };
      };
    };
}