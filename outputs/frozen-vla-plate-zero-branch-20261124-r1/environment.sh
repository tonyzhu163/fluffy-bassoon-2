# Load the exact qualified eval runtime, then check this experiment's checkout.
zero_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
zero_repo=$(cd "$zero_here/../.." && pwd -P)
zero_pin=${ZERO_PIN:?new study git pin}
[[ $zero_pin =~ ^[0-9a-f]{40}$ && $(git -C "$zero_repo" rev-parse HEAD) == "$zero_pin" ]]
[[ -z $(git -C "$zero_repo" status --porcelain --untracked-files=no -- "$zero_here") ]]
old=/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-plate-transfer-20261124-r1
export EVAL_PIN=796acc7fa4ff8fb536eaa705a4e9d83c86cda0c3
qualified="$old/eval/code-$EVAL_PIN/outputs/frozen-vla-plate-transfer-20261124-r1/eval"
source "$qualified/environment.sh"
shared="$root/outputs/frozen-vla-plate-zero-branch-20261124-r1"
fit_pin=7f749ae21b8d74ad58536c00ae4aafd5ffca45c5
fit_sha=7d845ce26bbb7f7f5da45b4885660023dd0e33b622cb9913547d50290ca16fe2
fit="$old/fit/gpu03"
fit_tree="$old/fit/code-$fit_pin"
[[ $(git -C "$fit_tree" rev-parse HEAD) == "$fit_pin" && -z $(git -C "$fit_tree" status --porcelain --untracked-files=no) ]]
manifest="$old/eval/prepare01/panel/manifest.json"
manifest_sha=b7996a9e110c9ceed9babb081a18476c1dc0b9a8d7e623703f5912031e44bc3f
export PYTHONPATH="$zero_here:$qualified:$zero_repo/outputs/frozen-vla-plate-transfer-20261124-r1/stats:$fit_tree/outputs/frozen-vla-plate-transfer-20261124-r1/fit:$PYTHONPATH"
