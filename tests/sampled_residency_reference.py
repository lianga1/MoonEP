"""Pure NumPy structural oracle for configurable sampled-residency kernels.

No CUDA or unpublished workspace module is required. The oracle uses exact
Python tuples for frequency counting and Python integers for expert sets.
force_hash_collision=True makes both diagnostic hashes zero; exact tuple
counting and all decisions remain unchanged, including when hashes collide.
"""
from collections import Counter
import itertools
from fractions import Fraction
from numbers import Integral

import numpy as np


def reference(routes, owner, native_copies, sample_ids, *, experts,
              groups_per_class=2, force_hash_collision=False, candidate_limit=0):
    if isinstance(candidate_limit, bool) or not isinstance(candidate_limit, Integral) or not 0 <= candidate_limit <= 2**31-1:
        raise ValueError("candidate_limit must be an integer in 0..2**31-1")
    routes, owner, copies = map(np.asarray, (routes, owner, native_copies))
    ids = np.asarray(sample_ids)
    R, S, K = routes.shape
    E, D, G, B = int(experts), len(ids), int(groups_per_class), copies.shape[1]
    if R != 8 or not 8 <= E <= 4096 or E % R or not 1 <= K <= min(32, E):
        raise ValueError("invalid R/E/K")
    if not 1 <= D <= min(4096, R*S) or not 1 <= G <= 8 or not 1 <= B <= E-E//R:
        raise ValueError("invalid D/G/B")
    assert owner.shape == routes.shape and copies.shape == (R, B)
    assert np.all((routes >= 0) & (routes < E)) and np.all((owner >= 0) & (owner < R))
    assert np.all(np.diff(np.sort(routes, axis=-1), axis=-1) > 0)
    assert np.all((ids >= 0) & (ids < R*S)) and np.all(np.diff(ids) > 0)
    words, stride, classes_per_pair = (E+63)//64, K+1, 4*(K+1)
    buckets, retained = R*R*stride, 2*G
    home = np.arange(E)//(E//R)
    support = home[None, :] == np.arange(R)[:, None]
    for rank in range(R):
        valid = copies[rank][copies[rank] >= 0]
        assert np.all(valid < E) and len(set(valid.tolist())) == len(valid)
        assert np.all(home[valid] != rank)
        support[rank, valid] = True
    assert np.all(support[owner, routes]), "native copies do not cover the seed"
    extra = (support & (home[None, :] != np.arange(R)[:, None])).sum(1).astype(np.int32)
    initial_support = support.copy()
    pairs = list(itertools.combinations(range(R), 2))
    flat, own = routes.reshape(-1, K), owner.reshape(-1, K)
    classes = np.full((len(pairs), D), -1, np.int32)
    groups = np.full((len(pairs), D, K), -1, np.int32)
    sig0, sig1 = np.zeros((len(pairs), D), np.uint64), np.zeros((len(pairs), D), np.uint64)
    masks = np.zeros((len(pairs), D, words), np.uint64)
    group_tuples, bitmasks = {}, {}
    full_word = (1 << 64)-1
    def mix64(value):
        value = ((value ^ (value >> 30)) * 0xbf58476d1ce4e5b9) & full_word
        value = ((value ^ (value >> 27)) * 0x94d049bb133111eb) & full_word
        return value ^ (value >> 31)
    for p, (a, b) in enumerate(pairs):
        for j, token in enumerate(ids):
            source = int(token//S)
            ga = tuple(sorted(int(e) for e, o in zip(flat[token], own[token]) if o == a))
            gb = tuple(sorted(int(e) for e, o in zip(flat[token], own[token]) if o == b))
            choices = []
            for direction, (old, new, group, reverse) in enumerate(((a,b,ga,gb),(b,a,gb,ga))):
                gain = int(old != source)-int(new != source and not reverse)
                if group and gain >= 0 and sum(not support[new,e] for e in group) <= B-extra[new]:
                    choices.append((gain, direction, group))
            if not choices:
                continue
            maxgain = max(row[0] for row in choices)
            choices = [row for row in choices if row[0] == maxgain]
            tie = (int(token)+a*0x9e3779b9+b*0x85ebca6b) & 0xffffffff
            tie ^= tie >> 16
            tie = (tie*0x7feb352d) & 0xffffffff
            tie ^= tie >> 15
            gain, direction, group = choices[tie % len(choices)]
            classes[p,j] = (direction*stride+len(group))*2+gain
            groups[p,j,:len(group)] = group
            record = p*D+j
            group_tuples[record] = group
            bitmasks[record] = sum(1 << e for e in group)
            h, other = 0xcbf29ce484222325, 0x9e3779b97f4a7c15
            for e in group:
                h = ((h ^ (e+1))*0x100000001b3) & full_word
                other = mix64(other ^ (e+1))
                masks[p,j,e//64] |= np.uint64(1) << np.uint64(e%64)
            if not force_hash_collision:
                sig0[p,j], sig1[p,j] = h, other
    frequencies = np.zeros_like(classes)
    tops = np.full((len(pairs), classes_per_pair, G), -1, np.int32)
    first = np.full((len(pairs), classes_per_pair), D, np.int32)
    for p in range(len(pairs)):
        occurrences = {}
        for j in range(D):
            if classes[p,j] >= 0:
                occurrences.setdefault((int(classes[p,j]), group_tuples[p*D+j]), []).append(j)
        for indices in occurrences.values():
            frequencies[p,indices[0]] = len(indices)
        for c in range(classes_per_pair):
            matches = np.flatnonzero(classes[p] == c)
            if len(matches):
                first[p,c] = matches[0]
            leaders = [p*D+int(j) for j in matches if frequencies[p,j] > 0]
            leaders.sort(key=lambda rec: (-int(frequencies.ravel()[rec]), group_tuples[rec]))
            tops[p,c,:min(G,len(leaders))] = leaders[:G]
    records, sizes = np.full((buckets, retained), -1, np.int32), np.zeros(buckets, np.int32)
    offsets, packages = [], []
    for p, (a,b) in enumerate(pairs):
        for size in range(1,K+1):
            offsets.append(len(packages))
            ka, kb = (a*R+b)*stride+size, (b*R+a)*stride+size
            for direction, key in ((0,ka),(1,kb)):
                gains = sorted((0,1), key=lambda gain: int(first[p,(direction*stride+size)*2+gain]))
                selected = [int(rec) for gain in gains for rec in tops[p,(direction*stride+size)*2+gain] if rec >= 0]
                sizes[key] = len(selected)
                records[key,:len(selected)] = selected
            for ra in records[ka,:sizes[ka]]:
                for rb in records[kb,:sizes[kb]]:
                    if (classes.ravel()[ra]&1)+(classes.ravel()[rb]&1):
                        packages.append((a,b,size,int(ra),int(rb)))
    offsets.append(len(packages))
    resident = [sum(1 << int(e) for e in np.flatnonzero(row)) for row in support]
    history, history_rows = [], []
    def enabled_counts(key,bits):
        values = [0,0]
        for rec in records[key,:sizes[key]]:
            if bitmasks[rec] & ~bits == 0:
                values[classes.ravel()[rec]&1] += int(frequencies.ravel()[rec])
        return values
    def match(x,y):
        n = min(sum(x),sum(y),max(x[1],y[1]))
        return min(n,x[1])+min(n,y[1])
    selected_ids = list(range(len(packages)))
    if candidate_limit:
        initial = [enabled_counts(key,resident[(key//stride)%R]) for key in range(buckets)]
        def initial_score(index):
            a,b,size,ra,rb = packages[index]
            aa,ab = bitmasks[rb]&~resident[a], bitmasks[ra]&~resident[b]
            ca,cb = bin(aa).count("1"),bin(ab).count("1")
            cost = ca+cb
            if cost == 0 or extra[a]+ca > B or extra[b]+cb > B:
                return Fraction(0),0,0,-index
            ka,kb = (a*R+b)*stride+size,(b*R+a)*stride+size
            gain = match(enabled_counts(ka,resident[b]|ab),enabled_counts(kb,resident[a]|aa))-match(initial[ka],initial[kb])
            return (Fraction(gain,cost),gain,-cost,-index) if gain > 0 else (Fraction(0),0,0,-index)
        selected_ids.sort(key=initial_score,reverse=True)
        selected_ids = selected_ids[:candidate_limit]
    retained_ids = sorted(selected_ids)
    while True:
        enabled = [enabled_counts(key,resident[(key//stride)%R]) for key in range(buckets)]
        best = None
        for index in retained_ids:
            a,b,size,ra,rb = packages[index]
            aa, ab = bitmasks[rb]&~resident[a], bitmasks[ra]&~resident[b]
            ca, cb = bin(aa).count("1"), bin(ab).count("1")
            cost = ca+cb
            if cost == 0 or extra[a]+ca > B or extra[b]+cb > B:
                continue
            ka, kb = (a*R+b)*stride+size, (b*R+a)*stride+size
            gain = match(enabled_counts(ka,resident[b]|ab),enabled_counts(kb,resident[a]|aa))-match(enabled[ka],enabled[kb])
            if gain <= 0:
                continue
            score = Fraction(gain,cost),gain,-cost,-index
            if best is None or score > best[0]:
                best = score,index,a,b,size,gain,aa,ab,ca,cb
        if best is None:
            break
        _,index,a,b,size,gain,aa,ab,ca,cb = best
        resident[a] |= aa
        resident[b] |= ab
        extra[a] += ca
        extra[b] += cb
        history.append(dict(a=a,b=b,add_a=[e for e in range(E) if (aa>>e)&1],
            add_b=[e for e in range(E) if (ab>>e)&1],size=size,estimated_sample_gain=gain))
        history_rows.append([index,gain,a,b,size,ca+cb]+
            [(bits>>(64*w))&full_word for bits in (aa,ab) for w in range(words)])
    result = np.array([[(bits>>e)&1 for e in range(E)] for bits in resident], dtype=bool)
    return dict(support=result, extra=extra, initial_support=initial_support,
        classes=classes, groups=groups, sig0=sig0, sig1=sig1, masks=masks,
        frequencies=frequencies, tops=tops, first=first, records=records,
        sizes=sizes, package_offsets=np.asarray(offsets,np.int32),
        packages=np.asarray(packages,np.int32).reshape(-1,5),
        history=history, history_rows=history_rows,
        selected_ids=np.asarray(selected_ids,np.int32), retained_ids=np.asarray(retained_ids,np.int32))
