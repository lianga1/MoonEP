from concurrent.futures import ThreadPoolExecutor
from moonep.sampled_residency import extension
cases=[(1024,7,128,8),(520,16,512,3),(1024,32,128,8),(512,16,128,2)]
def build(case):
 print('compile',case,flush=True)
 extension(*case)
 print('ready',case,flush=True)
with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(build,cases))
print('Remaining extensions ready',flush=True)
