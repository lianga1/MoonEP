"""Make a self-contained, path-scoped MoonEP experiment delivery without Git changes."""
from pathlib import Path
import argparse,hashlib,json,re,shutil

ROOT=Path(__file__).resolve().parents[2]
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
DEST=args.output.resolve()
if DEST.exists():raise FileExistsError(DEST)
DEST.mkdir(parents=True)
copied={};omitted=[];changes=[]

def skip(path):
    return any(p in ('.git','__pycache__','.pytest_cache','build','.AppleDouble') or p.startswith('._') for p in path.parts) or path.name=='.DS_Store' or path.suffix in ('.pyc','.so','.o','.a','.nsys-rep','.ncu-rep')

def copy(path):
    path=Path(path)
    if not path.is_file() or skip(path):return
    rel=path.relative_to(ROOT);out=DEST/rel;out.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(path,out);copied[str(rel)]=hashlib.sha256(path.read_bytes()).hexdigest()

def tree(name):
    for p in sorted((ROOT/name).rglob('*')):copy(p)

for name in ('experiments/eight_rank_estimate','experiments/two_gpu_overlap','experiments/h100_planner',
             'inputs/MoonEP','outputs/eight_rank_estimate','outputs/two_gpu_overlap','outputs/two_gpu_training',
             'outputs/h100_planner','outputs/h100_subset_reference','outputs/h100_subset_validation',
             'outputs/aggregate_exchange_multipair','outputs/pruned_subset_indexed','outputs/planner_explainer','analysis'):
    tree(name)
# Keep current GPU references and all current raw failures, but not unrelated
# 4x8 simulation or every historical generated routing variant.
for p in (ROOT/'experiments/bucket8').glob('*'):
    if p.is_file():copy(p)
for name in ('analysis/moonep_cpu.py','analysis/vendor/residency.py','analysis/vendor/bucket_optimizer.py',
             'outputs/planner_explainer/sampling_evidence.json','outputs/rt2_b8_comparison/results.json',
             'outputs/rt2_b8_comparison/summary.json','outputs/rt2_b8_comparison/independent_audit.json',
             'outputs/trace_audit/summary.json','outputs/replay/small_sample.json',
             'outputs/aggregate_exchange/results.json','outputs/gpu/BUCKET8_REPORT_ZH.md',
             'outputs/gpu/NATIVE_OPTIMIZATION_REPORT_ZH.md'):
    copy(ROOT/name)
for layer in range(8):
    for name in (f'outputs/trace_audit/layer_{layer:02}.npz',
                 f'outputs/rt2_b8_comparison/layer{layer:02}_native_rt0.npz',
                 f'outputs/rt2_b8_comparison/layer{layer:02}_native_rt2.npz',
                 f'outputs/replay/layer{layer:02}_b8_small256_global_4.npz',
                 f'outputs/aggregate_exchange_multipair/layer{layer:02}_mixed_sizes.npz',
                 f'outputs/pruned_subset_indexed/layer{layer:02}_h2_g8.npz'):
        copy(ROOT/name)

def edit(name,before,after,reason):
    p=DEST/name;old=p.read_text();assert before in old,(name,before)
    p.write_text(old.replace(before,after));changes.append(dict(path=name,reason=reason))

edit('experiments/two_gpu_overlap/run_distributed.sh',
     'export PATH=/root/anaconda3/envs/sglang/bin:$PATH',
     '# Use the activated environment, optionally selecting its bin directory.\nif [[ -n "${MOONEP_PYTHON_BIN:-}" ]]; then export PATH="$MOONEP_PYTHON_BIN:$PATH"; fi',
     'Portable environment selection; CUDA kernels and timing protocol unchanged')
edit('experiments/eight_rank_estimate/prepare_kernel_inputs.py',
     'path=str(out),','path=str(out.relative_to(ROOT)),', 'New fixture manifests use experiment-relative paths')
edit('experiments/eight_rank_estimate/prepare_kernel_inputs.py',
     'original_sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest()',
     'original_sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()',
     'New fixture provenance uses experiment-relative paths')
edit('experiments/eight_rank_estimate/audit_kernel_results.py',
     'p=Path(source);ok=p.exists() and sha(p)==expected',
     'p=Path(source)\n            if p.is_absolute() and "/outputs/" in p.as_posix():\n                p=ROOT/"outputs"/p.as_posix().split("/outputs/",1)[1]\n            elif not p.is_absolute():\n                p=ROOT/p\n            ok=p.exists() and sha(p)==expected',
     'Resolve historical original-source paths inside the relocated experiment and retain SHA checks')
edit('experiments/eight_rank_estimate/audit_kernel_results.py',
     'import hashlib,json,math,statistics','import argparse,hashlib,json,math,statistics',
     'Add a separate audit output directory so published evidence remains immutable')
edit('experiments/eight_rank_estimate/audit_kernel_results.py',
     'def main():\n',
     'def main():\n    parser=argparse.ArgumentParser()\n    parser.add_argument("--output-dir",type=Path,default=OUT)\n    result_dir=parser.parse_args().output_dir\n    result_dir.mkdir(parents=True,exist_ok=True)\n',
     'Allow CPU readback into a fresh output directory')
edit('experiments/eight_rank_estimate/audit_kernel_results.py',
     "with (OUT/name).open('x') as f:","with (result_dir/name).open('x') as f:",
     'Write only newly requested audit outputs, preserving raw published results')

readme=ROOT/'outputs/eight_rank_estimate/github_README.md'
if not readme.exists():raise FileNotFoundError('README draft is required before publishing')
shutil.copyfile(readme,DEST/'README.md')
edit('README.md',
     '两卡带宽校准还需要按照 MoonEP 根目录安装说明编译并安装该 fork，使 `import moonep` 使用这份源码，以及可进行对称内存访问的两张互联 GPU。',
     '两卡带宽校准需要编译本包中测试过的 `inputs/MoonEP` 快照，并使用可进行对称内存访问的两张互联 GPU。实测 CuTe DSL 是 `nvidia-cutlass-dsl==4.5.2`（快照 setup.py 声明的是 4.4.2），因此在隔离的 CUDA Python 环境预装实测依赖，再执行 `python -m pip install --no-deps --no-build-isolation -e ./inputs/MoonEP`，避免安装时悄悄改换 DSL。原 setup.py 保留不改。本包脚本优先导入该快照，不能用父仓库其他版本的 `_C` 二进制替代。CPU 依赖固定为 NumPy 2.0.2 与 Numba 0.60.0，GPU 主环境为上表所列的 PyTorch/CUDA。',
     'Document the exact snapshot build required by native calibration')
edit('README.md',
     '跨目录使用时应解析清单中的历史路径到本地对应输入，并继续验证原 SHA256，不能用新输入替换旧证据。',
     '发布版已经把清单中的历史路径重定位到本包对应输入，并继续验证原 SHA256。用 `python experiments/eight_rank_estimate/audit_kernel_results.py --output-dir local_runs/readback` 将新审计写入独立目录；不要用新输入替换旧证据。`analysis/trace_audit.py --synthetic` 是历史对照入口，仍依赖原工作区额外的合成生成器；本包的真实路由与主复现流程不使用该选项。',
     'Document relocated CPU audit and out-of-scope optional synthetic fixture generator')
(DEST/'requirements-cpu.txt').write_text('numpy==2.0.2\nnumba==0.60.0\n')
(DEST/'.gitignore').write_text('__pycache__/\n*.pyc\n*.so\nbuild/\nlocal_runs/\n')
(DEST/'PUBLICATION.md').write_text('''# Publication and provenance

This directory preserves the original GPU-executed CUDA/Python modules and raw
timing, correctness, and failure evidence. Historical absolute paths in raw
evidence identify the measurement environment; they are not required runtime
locations. No credential material, model weights, original token text, compiled
extensions, Git object databases, or complete router-dump archives are included.
The NPZ fixtures contain expert IDs, execution-rank plans, and numerical audits.

`PACKAGE_MANIFEST.json` records original and published hashes. Portability-only
edits are listed separately; they do not rewrite the hashes recorded at GPU test
time or claim that a different kernel was measured. `inputs/MoonEP` is the tested
source snapshot (with its license), not a replacement for the parent fork API.
The optional DeepEP snapshots retain their licenses and recorded commits.

Only the documented entry points are part of this delivery. Legacy provenance
archives and logs retain historical paths and failed attempts. Source/archive
manifests from the original workspace remain historical evidence; the package
manifest is the current distribution inventory.
''')
files=[]
for p in sorted(DEST.rglob('*')):
    if p.is_file():
        rel=str(p.relative_to(DEST));sha=hashlib.sha256(p.read_bytes()).hexdigest()
        files.append(dict(path=rel,bytes=p.stat().st_size,original_sha256=copied.get(rel),published_sha256=sha,
                          unchanged_from_source=sha==copied.get(rel) if rel in copied else None))
manifest=dict(scope='Experimental R8 planner components, R2 physical calibration and eight-rank model, plus retained H100 comparisons',
              transformations=changes,files=files,total_bytes=sum(x['bytes'] for x in files),
              excluded=['.git and caches','compiled binaries','unrelated multinode simulation','complete raw input archives'])
(DEST/'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(output=str(DEST),files=len(files),MiB=manifest['total_bytes']/2**20,changed_paths=sorted(set(x['path'] for x in changes)))))
