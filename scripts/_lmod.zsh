# Make `module` usable inside a batch script. Sourced, never executed.
#
# `module` is a shell function rather than a program: Lmod defines it in an init
# script that a *login* shell sources. A SLURM batch script is not a login
# shell, so the function is simply absent and the first `module load ...` dies
# with `command not found: module` -- which under `set -e` takes the whole job
# with it, after the expensive part has already run.
#
# This did not bite while the scripts were bash. Bash exports functions through
# the environment (`BASH_FUNC_module%%`) and SLURM propagates the environment,
# so the function arrived for free. zsh does not import those, so moving the
# scripts to zsh is precisely what exposed it. Observed as
# `slurm_script:65: command not found: module`.
#
# `MODULESHOME` is an ordinary exported *variable*, so unlike the function it
# does survive into the job; it is the thing to trust, with Perlmutter's path as
# the fallback for the case where even that is missing.
#
# `whence` rather than `typeset -f`: it finds a command, builtin or alias too,
# so a site that ships `module` as something other than a function is left
# alone instead of being re-sourced over.

if ! whence module > /dev/null 2>&1; then
  # `MODULESHOME` first because it is what the site actually set. The rest are
  # fallbacks for the case where even that did not survive: the generic Lmod
  # prefix, then Cray's, which is where Perlmutter keeps it.
  lmod_candidates=(
    ${MODULESHOME:+"${MODULESHOME}/init/zsh"}
    /usr/share/lmod/lmod/init/zsh
    /opt/cray/pe/lmod/lmod/init/zsh
  )

  for lmod_init in "${lmod_candidates[@]}"; do
    if [[ -r "${lmod_init}" ]]; then
      source "${lmod_init}"
      break
    fi
  done

  if ! whence module > /dev/null 2>&1; then
    print -u2 "Cannot initialise the module system: \`module\` is undefined and"
    print -u2 "none of these were readable:"
    print -u2 ${(F)lmod_candidates}
    print -u2 "Check \`echo \$MODULESHOME\` on a login node and export it."
    exit 1
  fi
fi
